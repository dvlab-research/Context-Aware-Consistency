"""
Example:
``` py
model = torch.hub.load('Ivan1248/Context-Aware-Consistency', 'DeepLabV3Plus', backbone='resnet50', num_classes=19, pretrained=True) 
```
"""

from models.modeling.deeplab import DeepLab as DeepLabV3Plus
from models import CAC
