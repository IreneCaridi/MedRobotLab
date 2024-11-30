import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import cv2
from . import random_state

random_state()


def obj2lab(obj):
    if obj != 'bkg':
        return 1
    else:
        return 0


def int2color(integer, max_value=10):
    """
    Convert an integer to a distinct color using a colormap.

    Args:
        - integer: int, the integer to convert to a color.
        - max_value: int, the maximum possible value of the integer (for normalizing the color range).

    Returns:
           - color: str or tuple, the color suitable for plt.text.
    """
    if integer == 'bkg':
        integer = 0

    # Normalize the integer between 0 and 1 based on the max_value
    normalized_value = integer / max_value

    # Use a colormap (e.g., 'tab10', which has distinct colors) to map the integer to a color
    cmap = plt.get_cmap('tab10')  # You can choose different colormaps here (e.g., 'viridis', 'rainbow', etc.)

    # Get the color from the colormap
    color = cmap(normalized_value % 1)  # Using % 1 to keep values within the colormap range

    return color


# Function to open a file dialog to select images
def select_images():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_paths = filedialog.askopenfilenames(title="Select Images",
                                             filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
    return list(file_paths)


# Helper function to update the title with the current label
def update_title(target_obj):
    plt.title(f"Current Label: {target_obj}. Press right or left "
              "or b to switch labels, r to remove last.")
    plt.draw()


def annotate_image(image_path):
    """
        Function to display image and capture user points with labels

        args:
            -image_path: path to image
    """
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for displaying in matplotlib

    # Dictionary to store points for each label
    labeled_points = []

    # Current active label (starts with 'obj 1')
    target_obj = 1
    last = 1

    # Plotting setup
    fig, ax = plt.subplots()
    ax.imshow(image_rgb)

    plotted_points = []

    # Function to handle mouse clicks and store points with the current label
    def onclick(event):
        ix, iy = event.xdata, event.ydata
        if ix is not None and iy is not None:
            labeled_points.append(([int(ix), int(iy)], target_obj))
            print(f"Point selected: ({ix:.0f}, {iy:.0f}) for target {target_obj}")
            # Draw a small circle at the clicked point and label it
            scatter = plt.scatter(ix, iy, s=40, color=int2color(target_obj))
            text = plt.text(ix, iy, f"obj {target_obj}", fontsize=8, color=int2color(target_obj))

            plotted_points.append((scatter, text))
            plt.draw()

    # Function to handle key presses for switching labels
    def onkey(event):
        nonlocal target_obj
        nonlocal last
        nonlocal labeled_points
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
        elif event.key == 'r':
            print(f'Last point removed')

            # removing last point from list
            labeled_points = labeled_points[:-1]

            # removing last point from plot
            last_scatter, last_text = plotted_points.pop()
            last_scatter.remove()
            last_text.remove()
            plt.draw()

        update_title(target_obj)

    # Connect the click and key events
    cid_click = fig.canvas.mpl_connect('button_press_event', onclick)
    cid_key = fig.canvas.mpl_connect('key_press_event', onkey)

    # Initial title update
    update_title(target_obj)

    # Show the plot and wait for user interaction
    plt.show()

    # Return the points collected for each label
    return labeled_points


def get_batch_and_dataset_gui():

    print('waiting for user input...')

    # Create main window
    root = tk.Tk()
    root.title("Parameter Input")
    inputs = {"batch_size": None, "path": None}

    def submit():
        """Validate inputs and close the GUI if valid."""
        try:
            # Validate batch size (mandatory and integer)
            batch_size = int(entry_batch.get())
            inputs["batch_size"] = batch_size
        except ValueError:
            messagebox.showerror("Input Error", "Batch size must be an integer.")
            return

        # Collect paths, allow skipping only if at least one path is provided
        path = entry_path1.get().strip()

        inputs["path"] = path

        if not path:
            messagebox.showerror("Input Error", "At least one path must be provided.")
            return

        # If validation passes, close the GUI
        root.destroy()

    # Create labels and input fields
    tk.Label(root, text="Batch Size (mandatory):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
    entry_batch = tk.Entry(root)
    entry_batch.grid(row=0, column=1, padx=5, pady=5)

    tk.Label(root, text="Dataset Path:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
    entry_path1 = tk.Entry(root)
    entry_path1.grid(row=1, column=1, padx=5, pady=5)

    # Add submit button
    tk.Button(root, text="Submit", command=submit).grid(row=4, column=0, columnspan=2, pady=10)

    # Run the GUI
    root.mainloop()

    return inputs

