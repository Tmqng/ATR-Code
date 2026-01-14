import timm
from absl import app, logging
from torchinfo import summary


def create_swin_model(config):    
    """
    Create a Swin Transformer model adapted to SAR.

    Args:
        config: Configuration

    Returns:
        model: Swin Transformer
    """
    model = timm.create_model(
        config["model_name"],
        pretrained=config["pretrained"],
        num_classes=config["num_classes"],
        img_size=config["img_size"],
        in_chans=1,  # Grayscale SAR
        drop_rate=config["dropout"],
        drop_path_rate=config["drop_path"],
    )

    logging.info("Model created:")
    logging.info(summary(
            model,
            input_size=(1, 1, config["img_size"], config["img_size"]),
            device=model.device,
            verbose=0,
        ))

    return model


def main(_):
    # Example configuration
    cfg = {
        "model_name": "swin_tiny_patch4_window7_224",
        "pretrained": True,
        "num_classes": 10,
        "img_size": 224,
        "dropout": 0.1,
        "drop_path": 0.1,
        "device": "cpu",
    }

    _ = create_swin_model(cfg)


if __name__ == "__main__":
    app.run(main)