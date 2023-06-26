import os
import tkinter as tk
from PIL import Image, ImageTk

# Directory containing the images
image_dir = "images/dummy_data"

class ImageGallery:
    """
    A simple image gallery that displays a list of images in a directory
    """
    def __init__(self, root):
        """
        Create the image gallery
        """
        self.root = root
        self.thumbnails_frame = None
        self.enlarged_image_label = None
        self.images = []

        # Create the main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create a canvas for the thumbnails frame
        canvas = tk.Canvas(main_frame, height=100)
        canvas.pack(side=tk.BOTTOM, fill=tk.X)

        # Create a scrollbar for the canvas
        scrollbar = tk.Scrollbar(main_frame, orient=tk.HORIZONTAL, command=canvas.xview)
        scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Configure the canvas to work with the scrollbar
        canvas.configure(xscrollcommand=scrollbar.set)
        canvas.bind('<Configure>', lambda event: canvas.configure(scrollregion=canvas.bbox('all')))

        # Create the thumbnails frame inside the canvas
        self.thumbnails_frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=self.thumbnails_frame, anchor='nw')

        # Create the enlarged image label
        self.enlarged_image_label = tk.Label(main_frame)
        self.enlarged_image_label.pack(fill=tk.BOTH, expand=True)

        # Load and display the images
        self.load_thumbnail_images()

    def load_thumbnail_images(self):
        """
        Load the images from the directory and display them in the thumbnails frame
        """
        # Get the list of image files in the directory
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            if os.path.isfile(image_path):
                # Load the image and create a thumbnail
                image = Image.open(image_path)
                thumbnail = image.copy()
                thumbnail.thumbnail((100, 100))

                # Convert the thumbnail to Tkinter-compatible format
                thumbnail_tk = ImageTk.PhotoImage(thumbnail)

                # Create a label with the thumbnail image
                thumbnail_label = tk.Label(self.thumbnails_frame, image=thumbnail_tk)
                thumbnail_label.image = thumbnail_tk  # Store a reference to prevent garbage collection
                thumbnail_label.pack(side=tk.LEFT, padx=5)
                # Bind the label to display the enlarged image
                thumbnail_label.bind("<Button-1>", lambda event, img=image: self.display_enlarged_image(img))

                # Add the image and its label to the list
                self.images.append((image, thumbnail_label))

    def display_enlarged_image(self, image):
        """
        Display the image in the enlarged image label while maintaining aspect ratio and fitting window height
        """
        # Get the size of the window
        window_width = self.root.winfo_width()
        window_height = self.root.winfo_height() - self.thumbnails_frame.winfo_height()

        # Get the size of the image
        image_width, image_height = image.size

        # Calculate the aspect ratio of the image
        aspect_ratio = image_width / image_height

        # Calculate the new width and height to fit the window height while maintaining aspect ratio
        new_height = window_height
        new_width = int(new_height * aspect_ratio)

        # Resize the image with the calculated size
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)

        # Convert the resized image to Tkinter-compatible format
        resized_image_tk = ImageTk.PhotoImage(resized_image)

        # Update the enlarged image label
        self.enlarged_image_label.configure(image=resized_image_tk)
        self.enlarged_image_label.image = resized_image_tk


# Create the main window
root = tk.Tk()
root.title("Image Gallery")
root.geometry("1000x800")

# Create the image gallery
gallery = ImageGallery(root)

# Run the GUI main loop
root.mainloop()
