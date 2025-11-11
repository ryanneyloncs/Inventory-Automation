import asyncio
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
from src.models.database import (
    Product, InventoryLevel, OptimizationParameters,
    PurchaseOrder, PurchaseOrderItem, Supplier, AlertLog
)
from src.services.suppliers.supplier_client import SupplierAPIClient
from config.settings import settings
import logging
from decimal import Decimal

logger = logging.getLogger(__name__)


class ReorderLogic:
    """Core reorder decision logic"""
    
    def __init__(self):
        self.auto_reorder_enabled = settings.AUTO_REORDER_ENABLED
        self.min_order_qty = settings.MIN_ORDER_QUANTITY
        self.max_order_qty = settings.MAX_ORDER_QUANTITY
    
    def should_reorder(
        self,
        current_quantity: int,
        reorder_point: int,
        quantity_on_order: int = 0
    ) -> bool:
        """
        Determine if product should be reordered
        
        Reorder when: Current Quantity + Quantity On Order <= Reorder Point
        """
        available_quantity = current_quantity + quantity_on_order
        return available_quantity <= reorder_point
    
    def calculate_order_quantity(
        self,
        current_quantity: int,
        reorder_point: int,
        eoq: int,
        max_stock_level: int,
        supplier_min_qty: int = 1
    ) -> int:
        """
        Calculate optimal order quantity
        
        Strategy:
        1. Order EOQ if it brings us between reorder point and max stock
        2. Adjust if we'd exceed max stock
        3. Respect supplier minimum order quantity
        """
        # How much do we need to reach max stock?
        quantity_needed = max_stock_level - current_quantity
        
        # Use EOQ as base
        order_qty = eoq
        
        # Don't exceed max stock level
        if current_quantity + order_qty > max_stock_level:
            order_qty = max(quantity_needed, supplier_min_qty)
        
        # Ensure minimum order quantity
        order_qty = max(order_qty, supplier_min_qty)
        
        # Respect system limits
        order_qty = min(max(order_qty, self.min_order_qty), self.max_order_qty)
        
        return order_qty
    
    def calculate_urgency_score(
        self,
        current_quantity: int,
        reorder_point: int,
        avg_daily_demand: float,
        lead_time_days: int
    ) -> float:
        """
        Calculate urgency score for prioritizing orders
        
        Returns 0-100 score (higher = more urgent)
        """
        if avg_daily_demand <= 0:
            return 0
        
        # Days of inventory remaining
        days_remaining = current_quantity / avg_daily_demand if avg_daily_demand > 0 else 999
        
        # Compare to lead time
        urgency_ratio = 1 - (days_remaining / lead_time_days)
        urgency_ratio = max(0, min(1, urgency_ratio))
        
        # How far below reorder point?
        shortage_ratio = max(0, (reorder_point - current_quantity) / reorder_point) if reorder_point > 0 else 0
        
        # Combine metrics
        urgency_score = (urgency_ratio * 0.6 + shortage_ratio * 0.4) * 100
        
        return round(urgency_score, 2)


class ReorderService:
    """Main reorder service with full automation"""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.reorder_logic = ReorderLogic()
        self.supplier_client = SupplierAPIClient()
    
    async def check_reorder_needs(self) -> List[Dict]:
        """Scan all products and identify reorder needs"""
        logger.info("Checking reorder needs for all products")
        
        # Query products with their inventory and optimization parameters
        query = select(
            Product, InventoryLevel, OptimizationParameters, Supplier
        ).join(
            InventoryLevel, Product.id == InventoryLevel.product_id
        ).join(
            OptimizationParameters, Product.id == OptimizationParameters.product_id
        ).join(
            Supplier, Product.supplier_id == Supplier.id
        ).where(
            and_(
                Product.is_active == True,
                Supplier.is_active == True
            )
        )
        
        result = await self.db.execute(query)
        rows = result.all()
        
        reorder_candidates = []
        
        for row in rows:
            product = row.Product
            inventory = row.InventoryLevel
            params = row.OptimizationParameters
            supplier = row.Supplier
            
            # Calculate quantity already on order from pending POs
            quantity_on_order = await self._get_quantity_on_order(product.id)
            
            # Check if reorder is needed
            should_reorder = self.reorder_logic.should_reorder(
                current_quantity=inventory.quantity_available,
                reorder_point=params.reorder_point,
                quantity_on_order=quantity_on_order
            )
            
            if should_reorder:
                # Calculate order quantity
                order_qty = self.reorder_logic.calculate_order_quantity(
                    current_quantity=inventory.quantity_available,
                    reorder_point=params.reorder_point,
                    eoq=params.economic_order_quantity,
                    max_stock_level=params.max_stock_level,
                    supplier_min_qty=supplier.minimum_order_quantity
                )
                
                # Calculate urgency
                avg_daily_demand = params.demand_std_dev  # Simplified
                urgency = self.reorder_logic.calculate_urgency_score(
                    current_quantity=inventory.quantity_available,
                    reorder_point=params.reorder_point,
                    avg_daily_demand=avg_daily_demand,
                    lead_time_days=params.lead_time_days
                )
                
                reorder_candidates.append({
                    'product_id': product.id,
                    'sku': product.sku,
                    'name': product.name,
                    'supplier_id': supplier.id,
                    'supplier_name': supplier.name,
                    'current_quantity': inventory.quantity_available,
                    'reorder_point': params.reorder_point,
                    'order_quantity': order_qty,
                    'unit_cost': float(product.unit_cost),
                    'total_cost': order_qty * float(product.unit_cost),
                    'urgency_score': urgency,
                    'lead_time_days': params.lead_time_days
                })
        
        # Sort by urgency
        reorder_candidates.sort(key=lambda x: x['urgency_score'], reverse=True)
        
        logger.info(f"Found {len(reorder_candidates)} products needing reorder")
        
        return reorder_candidates
    
    async def create_purchase_order(
        self,
        supplier_id: int,
        items: List[Dict],
        auto_send: bool = True
    ) -> Dict:
        """
        Create a purchase order
        
        Items format: [{'product_id': 1, 'quantity': 100}, ...]
        """
        logger.info(f"Creating purchase order for supplier {supplier_id}")
        
        # Get supplier
        supplier_query = select(Supplier).where(Supplier.id == supplier_id)
        supplier_result = await self.db.execute(supplier_query)
        supplier = supplier_result.scalar_one_or_none()
        
        if not supplier:
            return {
                'success': False,
                'error': 'Supplier not found'
            }
        
        # Generate order number
        order_number = await self._generate_order_number()
        
        # Calculate expected delivery
        expected_delivery = datetime.utcnow() + timedelta(days=supplier.lead_time_days)
        
        # Create PO
        po = PurchaseOrder(
            order_number=order_number,
            supplier_id=supplier_id,
            order_date=datetime.utcnow(),
            expected_delivery_date=expected_delivery,
            status='pending',
            total_amount=Decimal('0.00'),
            created_by='system'
        )
        
        self.db.add(po)
        await self.db.flush()  # Get PO ID
        
        # Add items
        total_amount = Decimal('0.00')
        
        for item in items:
            # Get product
            product_query = select(Product).where(Product.id == item['product_id'])
            product_result = await self.db.execute(product_query)
            product = product_result.scalar_one_or_none()
            
            if not product:
                continue
            
            quantity = item['quantity']
            unit_price = product.unit_cost
            line_total = unit_price * quantity
            
            po_item = PurchaseOrderItem(
                purchase_order_id=po.id,
                product_id=product.id,
                quantity_ordered=quantity,
                unit_price=unit_price,
                line_total=line_total
            )
            
            self.db.add(po_item)
            total_amount += line_total
        
        # Update PO total
        po.total_amount = total_amount
        
        await self.db.commit()
        
        # Send to supplier if auto-send enabled
        if auto_send and supplier.api_endpoint:
            try:
                send_result = await self.supplier_client.send_purchase_order(
                    supplier=supplier,
                    order_data={
                        'order_number': order_number,
                        'items': items,
                        'total_amount': float(total_amount),
                        'expected_delivery': expected_delivery.isoformat()
                    }
                )
                
                if send_result['success']:
                    po.status = 'sent'
                    await self.db.commit()
                    logger.info(f"PO {order_number} sent to supplier successfully")
                else:
                    logger.error(f"Failed to send PO to supplier: {send_result.get('error')}")
            
            except Exception as e:
                logger.error(f"Error sending PO to supplier: {e}")
        
        # Create alert
        await self._create_alert(
            alert_type='reorder_triggered',
            severity='info',
            message=f"Purchase order {order_number} created for supplier {supplier.name}",
            metadata={'po_id': po.id, 'total_amount': float(total_amount)}
        )
        
        logger.info(f"Purchase order {order_number} created successfully")
        
        return {
            'success': True,
            'purchase_order': {
                'id': po.id,
                'order_number': order_number,
                'supplier': supplier.name,
                'total_amount': float(total_amount),
                'status': po.status,
                'expected_delivery': expected_delivery.isoformat()
            }
        }
    
    async def process_automatic_reorders(self) -> Dict:
        """
        Automatically process reorders for all products
        This is the main automated workflow
        """
        if not self.reorder_logic.auto_reorder_enabled:
            logger.warning("Automatic reordering is disabled")
            return {
                'success': False,
                'error': 'Automatic reordering is disabled'
            }
        
        logger.info("Starting automatic reorder process")
        
        # Get reorder candidates
        candidates = await self.check_reorder_needs()
        
        if not candidates:
            logger.info("No products need reordering")
            return {
                'success': True,
                'total': 0,
                'purchase_orders_created': 0
            }
        
        # Group by supplier
        orders_by_supplier = {}
        for candidate in candidates:
            supplier_id = candidate['supplier_id']
            if supplier_id not in orders_by_supplier:
                orders_by_supplier[supplier_id] = []
            
            orders_by_supplier[supplier_id].append({
                'product_id': candidate['product_id'],
                'quantity': candidate['order_quantity']
            })
        
        # Create POs for each supplier
        created_pos = []
        total_successful = 0
        total_failed = 0
        
        for supplier_id, items in orders_by_supplier.items():
            try:
                result = await self.create_purchase_order(
                    supplier_id=supplier_id,
                    items=items,
                    auto_send=True
                )
                
                if result['success']:
                    created_pos.append(result['purchase_order'])
                    total_successful += 1
                else:
                    total_failed += 1
                    logger.error(f"Failed to create PO for supplier {supplier_id}")
            
            except Exception as e:
                logger.error(f"Error creating PO for supplier {supplier_id}: {e}")
                total_failed += 1
        
        logger.info(
            f"Automatic reorder completed: {total_successful} successful, "
            f"{total_failed} failed"
        )
        
        return {
            'success': True,
            'total': len(candidates),
            'successful': total_successful,
            'failed': total_failed,
            'purchase_orders': created_pos
        }
    
    async def receive_purchase_order(
        self,
        po_id: int,
        items_received: List[Dict]
    ) -> Dict:
        """
        Process received purchase order and update inventory
        
        items_received format: [{'product_id': 1, 'quantity': 95}, ...]
        """
        logger.info(f"Processing receipt for PO {po_id}")
        
        # Get PO
        po_query = select(PurchaseOrder).where(PurchaseOrder.id == po_id)
        po_result = await self.db.execute(po_query)
        po = po_result.scalar_one_or_none()
        
        if not po:
            return {
                'success': False,
                'error': 'Purchase order not found'
            }
        
        # Update PO items
        for item_data in items_received:
            item_query = select(PurchaseOrderItem).where(
                and_(
                    PurchaseOrderItem.purchase_order_id == po_id,
                    PurchaseOrderItem.product_id == item_data['product_id']
                )
            )
            item_result = await self.db.execute(item_query)
            po_item = item_result.scalar_one_or_none()
            
            if po_item:
                po_item.quantity_received = item_data['quantity']
                
                # Update inventory
                inv_query = select(InventoryLevel).where(
                    InventoryLevel.product_id == item_data['product_id']
                )
                inv_result = await self.db.execute(inv_query)
                inventory = inv_result.scalar_one_or_none()
                
                if inventory:
                    inventory.quantity_on_hand += item_data['quantity']
                    inventory.quantity_available = inventory.quantity_on_hand - inventory.quantity_reserved
                    inventory.last_counted_at = datetime.utcnow()
        
        # Update PO status
        po.status = 'received'
        po.actual_delivery_date = datetime.utcnow()
        
        await self.db.commit()
        
        logger.info(f"PO {po_id} received and inventory updated")
        
        return {
            'success': True,
            'message': f'Purchase order {po.order_number} received'
        }
    
    async def check_overdue_orders(self) -> List[Dict]:
        """Check for overdue purchase orders"""
        current_date = datetime.utcnow()
        
        query = select(PurchaseOrder, Supplier).join(
            Supplier, PurchaseOrder.supplier_id == Supplier.id
        ).where(
            and_(
                PurchaseOrder.status.in_(['pending', 'sent', 'confirmed']),
                PurchaseOrder.expected_delivery_date < current_date
            )
        )
        
        result = await self.db.execute(query)
        rows = result.all()
        
        overdue_orders = []
        
        for row in rows:
            po = row.PurchaseOrder
            supplier = row.Supplier
            
            days_overdue = (current_date - po.expected_delivery_date).days
            
            overdue_orders.append({
                'order_number': po.order_number,
                'supplier': supplier.name,
                'expected_delivery': po.expected_delivery_date.isoformat(),
                'days_overdue': days_overdue,
                'status': po.status
            })
            
            # Create alert for overdue orders
            await self._create_alert(
                alert_type='delivery_delay',
                severity='warning',
                message=f"PO {po.order_number} is {days_overdue} days overdue",
                metadata={'po_id': po.id, 'supplier_id': supplier.id}
            )
        
        if overdue_orders:
            logger.warning(f"Found {len(overdue_orders)} overdue purchase orders")
        
        return overdue_orders
    
    async def _generate_order_number(self) -> str:
        """Generate unique order number"""
        # Get count of orders today
        today = datetime.utcnow().date()
        query = select(func.count(PurchaseOrder.id)).where(
            func.date(PurchaseOrder.order_date) == today
        )
        result = await self.db.execute(query)
        count = result.scalar()
        
        # Format: PO-YYYYMMDD-XXXX
        order_number = f"PO-{datetime.utcnow().strftime('%Y%m%d')}-{count+1:04d}"
        
        return order_number
    
    async def _create_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        product_id: Optional[int] = None,
        metadata: Optional[Dict] = None
    ):
        """Create system alert"""
        alert = AlertLog(
            alert_type=alert_type,
            severity=severity,
            product_id=product_id,
            message=message,
            metadata=metadata
        )
        self.db.add(alert)
        await self.db.commit()
    
    async def get_low_stock_alerts(self) -> List[Dict]:
        """Get products with low stock levels"""
        query = select(
            Product, InventoryLevel, OptimizationParameters
        ).join(
            InventoryLevel, Product.id == InventoryLevel.product_id
        ).join(
            OptimizationParameters, Product.id == OptimizationParameters.product_id
        ).where(
            Product.is_active == True
        )
        
        result = await self.db.execute(query)
        rows = result.all()
        
        low_stock_products = []
        
        for row in rows:
            product = row.Product
            inventory = row.InventoryLevel
            params = row.OptimizationParameters
            
            # Check if below safety stock
            if inventory.quantity_available < params.safety_stock:
                low_stock_products.append({
                    'product_id': product.id,
                    'sku': product.sku,
                    'name': product.name,
                    'current_quantity': inventory.quantity_available,
                    'safety_stock': params.safety_stock,
                    'shortage': params.safety_stock - inventory.quantity_available
                })
        
        return low_stock_products
    
    async def _get_quantity_on_order(self, product_id: int) -> int:
        """
        Calculate total quantity on order for a product from pending POs
        
        Sums up quantities from all purchase orders that are:
        - Pending, sent, or confirmed (not yet received)
        - For the specified product
        """
        from sqlalchemy import func
        
        query = select(
            func.coalesce(func.sum(PurchaseOrderItem.quantity_ordered - PurchaseOrderItem.quantity_received), 0)
        ).join(
            PurchaseOrder, PurchaseOrderItem.purchase_order_id == PurchaseOrder.id
        ).where(
            and_(
                PurchaseOrderItem.product_id == product_id,
                PurchaseOrder.status.in_(['pending', 'sent', 'confirmed'])
            )
        )
        
        result = await self.db.execute(query)
        quantity_on_order = result.scalar() or 0
        
        return int(quantity_on_order)
