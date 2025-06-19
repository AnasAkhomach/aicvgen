from src.models.data_models import Item, ItemStatus

try:
    print('Testing Item creation...')
    item = Item(
        id='test',
        content='test content',
        status=ItemStatus.INITIAL
    )
    print('Item created successfully:', item)
except Exception as e:
    print('Error creating Item:', e)
    print('Error type:', type(e))