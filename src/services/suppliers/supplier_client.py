import asyncio
import aiohttp
from typing import Dict, Optional
from datetime import datetime
import logging
from config.settings import settings
from src.models.database import Supplier

logger = logging.getLogger(__name__)


class SupplierAPIClient:
    """Client for interacting with supplier APIs"""
    
    def __init__(self):
        self.timeout = settings.SUPPLIER_API_TIMEOUT
        self.retry_attempts = settings.SUPPLIER_API_RETRY_ATTEMPTS
        self.retry_delay = settings.SUPPLIER_API_RETRY_DELAY
    
    async def send_purchase_order(
        self,
        supplier: Supplier,
        order_data: Dict
    ) -> Dict:
        """
        Send purchase order to supplier API
        
        Args:
            supplier: Supplier database object
            order_data: Order information dict
        
        Returns:
            Dict with success status and response data
        """
        if not supplier.api_endpoint:
            return {
                'success': False,
                'error': 'Supplier has no API endpoint configured'
            }
        
        logger.info(f"Sending PO to supplier {supplier.name}")
        
        # Prepare request
        headers = self._build_headers(supplier)
        payload = self._format_order_payload(order_data)
        
        # Send with retries
        for attempt in range(self.retry_attempts):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url=f"{supplier.api_endpoint}/orders",
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as response:
                        
                        if response.status in [200, 201, 202]:
                            data = await response.json()
                            logger.info(f"PO sent successfully to {supplier.name}")
                            
                            return {
                                'success': True,
                                'supplier_order_id': data.get('order_id'),
                                'confirmation_number': data.get('confirmation_number'),
                                'estimated_delivery': data.get('estimated_delivery'),
                                'response': data
                            }
                        
                        elif response.status == 429:  # Rate limited
                            logger.warning(f"Rate limited by supplier {supplier.name}, retrying...")
                            await asyncio.sleep(self.retry_delay * (attempt + 1))
                            continue
                        
                        else:
                            error_text = await response.text()
                            logger.error(
                                f"Supplier API error {response.status}: {error_text}"
                            )
                            
                            if attempt < self.retry_attempts - 1:
                                await asyncio.sleep(self.retry_delay)
                                continue
                            
                            return {
                                'success': False,
                                'error': f"API returned {response.status}",
                                'details': error_text
                            }
            
            except asyncio.TimeoutError:
                logger.error(f"Timeout sending PO to {supplier.name} (attempt {attempt + 1})")
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay)
                    continue
                return {
                    'success': False,
                    'error': 'Request timeout'
                }
            
            except aiohttp.ClientError as e:
                logger.error(f"Client error sending PO: {e}")
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay)
                    continue
                return {
                    'success': False,
                    'error': str(e)
                }
            
            except Exception as e:
                logger.error(f"Unexpected error sending PO: {e}")
                return {
                    'success': False,
                    'error': str(e)
                }
        
        return {
            'success': False,
            'error': f'Failed after {self.retry_attempts} attempts'
        }
    
    async def check_order_status(
        self,
        supplier: Supplier,
        supplier_order_id: str
    ) -> Dict:
        """
        Check status of an order with supplier
        
        Args:
            supplier: Supplier database object
            supplier_order_id: Order ID in supplier's system
        
        Returns:
            Dict with order status information
        """
        if not supplier.api_endpoint:
            return {
                'success': False,
                'error': 'Supplier has no API endpoint configured'
            }
        
        headers = self._build_headers(supplier)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url=f"{supplier.api_endpoint}/orders/{supplier_order_id}",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        return {
                            'success': True,
                            'status': data.get('status'),
                            'tracking_number': data.get('tracking_number'),
                            'estimated_delivery': data.get('estimated_delivery'),
                            'actual_delivery': data.get('actual_delivery'),
                            'items': data.get('items', [])
                        }
                    else:
                        error_text = await response.text()
                        return {
                            'success': False,
                            'error': f"API returned {response.status}",
                            'details': error_text
                        }
        
        except Exception as e:
            logger.error(f"Error checking order status: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def get_product_availability(
        self,
        supplier: Supplier,
        product_skus: list
    ) -> Dict:
        """
        Check product availability with supplier
        
        Args:
            supplier: Supplier database object
            product_skus: List of SKUs to check
        
        Returns:
            Dict with availability information
        """
        if not supplier.api_endpoint:
            return {
                'success': False,
                'error': 'Supplier has no API endpoint configured'
            }
        
        headers = self._build_headers(supplier)
        payload = {'skus': product_skus}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url=f"{supplier.api_endpoint}/availability",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        return {
                            'success': True,
                            'availability': data.get('products', [])
                        }
                    else:
                        return {
                            'success': False,
                            'error': f"API returned {response.status}"
                        }
        
        except Exception as e:
            logger.error(f"Error checking availability: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def get_pricing(
        self,
        supplier: Supplier,
        product_skus: list,
        quantities: list
    ) -> Dict:
        """
        Get pricing information from supplier
        
        Args:
            supplier: Supplier database object
            product_skus: List of SKUs
            quantities: Corresponding quantities for each SKU
        
        Returns:
            Dict with pricing information
        """
        if not supplier.api_endpoint:
            return {
                'success': False,
                'error': 'Supplier has no API endpoint configured'
            }
        
        headers = self._build_headers(supplier)
        
        items = [
            {'sku': sku, 'quantity': qty}
            for sku, qty in zip(product_skus, quantities)
        ]
        
        payload = {'items': items}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url=f"{supplier.api_endpoint}/pricing",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        return {
                            'success': True,
                            'pricing': data.get('items', []),
                            'total': data.get('total'),
                            'currency': data.get('currency', 'USD')
                        }
                    else:
                        return {
                            'success': False,
                            'error': f"API returned {response.status}"
                        }
        
        except Exception as e:
            logger.error(f"Error getting pricing: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _build_headers(self, supplier: Supplier) -> Dict:
        """Build HTTP headers for API request"""
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': f'{settings.APP_NAME}/{settings.APP_VERSION}'
        }
        
        # Add API key if available
        if supplier.api_key:
            headers['Authorization'] = f'Bearer {supplier.api_key}'
        
        return headers
    
    def _format_order_payload(self, order_data: Dict) -> Dict:
        """Format order data for supplier API"""
        return {
            'order_number': order_data['order_number'],
            'items': order_data['items'],
            'total_amount': order_data['total_amount'],
            'expected_delivery': order_data['expected_delivery'],
            'shipping_address': {
                'company': settings.APP_NAME,
                'address': 'Your warehouse address',
                'city': 'Your city',
                'state': 'Your state',
                'zip': 'Your zip',
                'country': 'US'
            },
            'contact': {
                'email': settings.EMAIL_FROM,
                'phone': 'Your phone'
            },
            'timestamp': datetime.utcnow().isoformat()
        }


class MockSupplierClient(SupplierAPIClient):
    """Mock supplier client for testing without real API calls"""
    
    async def send_purchase_order(self, supplier: Supplier, order_data: Dict) -> Dict:
        """Mock implementation"""
        logger.info(f"MOCK: Sending PO to {supplier.name}")
        
        # Simulate API delay
        await asyncio.sleep(0.5)
        
        return {
            'success': True,
            'supplier_order_id': f"SUP-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            'confirmation_number': 'CONF-12345',
            'estimated_delivery': (datetime.utcnow() + timedelta(days=supplier.lead_time_days)).isoformat(),
            'response': {'message': 'Mock order accepted'}
        }
    
    async def check_order_status(self, supplier: Supplier, supplier_order_id: str) -> Dict:
        """Mock implementation"""
        return {
            'success': True,
            'status': 'confirmed',
            'tracking_number': 'TRACK-12345',
            'estimated_delivery': (datetime.utcnow() + timedelta(days=5)).isoformat()
        }
    
    async def get_product_availability(self, supplier: Supplier, product_skus: list) -> Dict:
        """Mock implementation"""
        return {
            'success': True,
            'availability': [
                {'sku': sku, 'available': True, 'quantity': 1000}
                for sku in product_skus
            ]
        }
    
    async def get_pricing(self, supplier: Supplier, product_skus: list, quantities: list) -> Dict:
        """Mock implementation"""
        return {
            'success': True,
            'pricing': [
                {'sku': sku, 'quantity': qty, 'unit_price': 10.00, 'total': qty * 10.00}
                for sku, qty in zip(product_skus, quantities)
            ],
            'total': sum(qty * 10.00 for qty in quantities),
            'currency': 'USD'
        }
