from matplotlib.widgets import RectangleSelector
import cv2
import matplotlib.pyplot as plt
from .active_graphic import update_title, int2color

def rectangle_capture(image_path):
    """
    Function to display image and capture user-drawn rectangles with labels.

    Args:
        - image_path: path to image

    Returns:
        - rectangle_coords: List of tuples containing rectangle data with label,
          in format [(x_center, y_center, width, height, label), ...]
    """
    image = cv2.imread(image_path)
    image_height, image_width = image.shape[:2]
    print(f"Image dimensions: Width: {image_width}, Height: {image_height}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for displaying in matplotlib

    rectangle_coords = []
    target_obj = 1
    last = 1

    fig, ax = plt.subplots()
    ax.imshow(image_rgb)

    def rectangle_data(x1, y1, x2, y2):
        x0, y0 = min(x1, x2), min(y1, y2)  # punto in alto a sinistra
        x1, y1 = max(x1, x2), max(y1, y2)  # punto in basso a destra
        return x0, y0, x1, y1


    def onselect(eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if x1 is not None and x2 is not None:
            x_0, y_0, x_1, y_1 = rectangle_data(x1, y1, x2, y2)
            rectangle_coords.append((x_0, y_0, x_1, y_1, target_obj))
            width = x_1 - x_0
            height = y_1 - y_0
            print(f"Rectangle with point top-left ({x_0:.5f}, {y_0:.5f}), width {width:.5f}, height {height:.5f} for target {target_obj}")

            rect = plt.Rectangle((x_0, y_0), width, height, edgecolor=int2color(target_obj), facecolor='none', lw=2)
            ax.add_patch(rect)
            plt.draw()

    def onkey(event):
        nonlocal target_obj
        nonlocal last
        if event.key == 'right':
            if target_obj != 'bkg':
                target_obj += 1
            else:
                target_obj = last
        elif event.key == 'left':
            if target_obj != 'bkg' and target_obj != 1:
                target_obj -= 1
            elif target_obj == 'bkg':
                target_obj = last
        elif event.key == 'b':
            last = target_obj
            target_obj = 'bkg'
        elif event.key == 'r' and rectangle_coords:
            print('Last rectangle removed')
            rectangle_coords.pop()
            ax.patches[-1].remove()
            plt.draw()

        update_title(target_obj)

    rect_selector = RectangleSelector(
        ax, onselect, useblit=True,
        button=[1],  # Left mouse button for rectangles
        minspanx=5, minspany=5, spancoords='pixels', interactive=True
    )

    cid_key = fig.canvas.mpl_connect('key_press_event', onkey)
    update_title(target_obj)
    plt.show()
    fig.canvas.mpl_disconnect(cid_key)

    return rectangle_coords

