import tensorflow as tf
from utils.dataset_generator import DatasetGenerator
from utils.plot_generator import plot_generator
from model.model_builder import base_model
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser()

# Set Convert to SavedMoel
parser.add_argument("--saved_model_path", type=str,   help="저장된 모델 가중치 경로",
                    default='./checkpoints/0712/_0712_B16_E100_LR-0.001_320-320_train-10%_UNet_best_loss.h5')
parser.add_argument("--batch_size",       type=int,    help="배치 사이즈값 설정",
                    default=16)
parser.add_argument("--num_classes",      type=int,    help="분류할 클래수 개수 설정",
                    default=1)
parser.add_argument("--image_size",       type=tuple,  help="조정할 이미지 크기 설정",
                    default=(320, 320))

# Set dataset directory path 
parser.add_argument("--dataset_dir",      type=str,    help="데이터셋 다운로드 디렉토리 설정",
                    default='./datasets/')


args = parser.parse_args()

dataset = DatasetGenerator(data_dir=args.dataset_dir, image_size=args.image_size, batch_size=args.batch_size)
model = base_model(image_size=args.image_size, output_channel=args.num_classes)
model.load_weights(args.saved_model_path)
valid_data = dataset.get_validData()

if __name__ == "__main__":

    # Set plot size
    rows = 1
    cols = 3

    for img, depth in valid_data:
        pred_depth = model.predict_on_batch(img)
    
        for i in range(args.batch_size):
            batch_img = img[i]
            batch_pred_depth = pred_depth[i]
            batch_depth = depth[i]
            
            
            fig = plt.figure()

            ax0 = fig.add_subplot(rows, cols, 1)
            ax0.imshow(batch_img)
            ax0.set_title('img')
            ax0.axis("off")

            ax0 = fig.add_subplot(rows, cols, 2)
            ax0.imshow(batch_pred_depth)
            ax0.set_title('pred depth')
            ax0.axis("off")

            ax0 = fig.add_subplot(rows, cols, 3)
            ax0.imshow(batch_depth)
            ax0.set_title('depth')
            ax0.axis("off")
        
            plt.show()



