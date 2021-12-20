import matplotlib.pyplot as plt

def plot_generator(img, gt, predict=None, rows=1, cols=3):
    rows = 1
    if predict is None:
        cols = 2

    fig = plt.figure()
    for i in range(len(img)):
        ax0 = fig.add_subplot(rows, cols, 1)
        ax0.imshow(img[i])
        ax0.set_title('Input image')
        ax0.axis("off")

        ax1 = fig.add_subplot(rows, cols, 2)
        ax1.imshow(gt[i])
        ax1.set_title('Groundtruth')
        ax1.axis("off")

        ax2 = fig.add_subplot(rows, cols, 3)
        ax2.imshow(predict[i])
        ax2.set_title('Prediction')
        ax2.axis("off")

        plt.show()

