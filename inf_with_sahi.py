from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image_as_pil

detection_model = AutoDetectionModel.from_pretrained(
        model_type='torchvision',
        model=model,
        confidence_threshold=args['threshold'],
        device=args['device'],
        category_mapping={str(i): CLASSES[i] for i in range(1, len(CLASSES))},
        # category_remapping={CLASSES[i]: i for i in range(1, len(CLASSES))}
    )