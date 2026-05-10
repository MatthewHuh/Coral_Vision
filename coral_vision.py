import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.feature import local_binary_pattern

def process_coral_image(image_path):
    # ---------------------------------------------------------
    # 1. Load the Image
    # ---------------------------------------------------------
    # OpenCV loads images in BGR format by default. We convert it to RGB 
    # so the colors look correct when we plot them with Matplotlib.
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ---------------------------------------------------------
    # Step 1: Grayscale Conversion
    # ---------------------------------------------------------
    # Convert to grayscale to simplify the image from 3 channels (RGB) to 1.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---------------------------------------------------------
    # Step 2: Thresholding (The Binary Mask)
    # ---------------------------------------------------------
    # We use an inverted threshold (THRESH_BINARY_INV) because we want the 
    # dark coral to become WHITE (the "foreground" object we want to find) 
    # and the bright white tiles to become BLACK (the background).
    # We also use OTSU's method, which automatically calculates the best threshold value.
    _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # ---------------------------------------------------------
    # Step 3: Noise Reduction (Morphology)
    # ---------------------------------------------------------
    # The thresholding might pick up the dark grid or tiny specks of algae.
    # We use morphological operations to clean this up.
    # A 'kernel' is essentially a little brush we use to clean the image.
    kernel = np.ones((5, 5), np.uint8)
    
    # 'Opening' removes tiny white noise (small specks) from the background
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    # 'Closing' fills in tiny black holes inside the white coral blobs
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

    # ---------------------------------------------------------
    # Step 4 & 5: Contour Detection and Final Measurement
    # ---------------------------------------------------------
    # Find the outlines (contours) of the white blobs in the cleaned mask.
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    # Create a copy of the original image so we can draw on it
    final_output = img_rgb.copy()
    
    for contour in contours:
        # Calculate the area of the contour in pixels
        area = cv2.contourArea(contour)

        # Filter out contours that are too small (noise) or too large (the black grid)
        # You will likely need to tweak these numbers (500 and 20000) based on your image!
        if 2800 < area < 350000:
            # Draw a bright green outline around the coral (thickness = 3)
            cv2.drawContours(final_output, [contour], -1, (255, 0, 0), 3)
            
            # Find the center of the contour to place the text
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                # Write the pixel area near the coral
                cv2.putText(final_output, f"{int(area)}px", (cX - 30, cY - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # ---------------------------------------------------------
    # Display the Storyboard using Matplotlib
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.canvas.manager.set_window_title('Coral Computer Vision Pipeline')

    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title('Step 1: Grayscale')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(cleaned_mask, cmap='gray')
    axes[1, 0].set_title('Steps 2 & 3: Binary Threshold & Shrink Expand Operations')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(final_output)
    axes[1, 1].set_title('Step 4: Final Contours & Pixel Area')
    axes[1, 1].axis('off')

    # ---------------------------------------------------------
    # Window 2: Display the Final Result
    # ---------------------------------------------------------
    # Create a new figure and axis for the final output
    fig2, ax_final = plt.subplots(figsize=(10, 8))
    fig2.canvas.manager.set_window_title('Coral Computer Vision Pipeline - Final Output')

    ax_final.imshow(final_output)
    ax_final.set_title('Step 4: Final Contours & Pixel Area')
    ax_final.axis('off')

    fig2.tight_layout()

    # ---------------------------------------------------------
    # Show all generated windows simultaneously
    # ---------------------------------------------------------

    fig.tight_layout()
    plt.show()

def process_coral_hsv(image_path):
    # 1. Load the Image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ---------------------------------------------------------
    # Step 1: Convert to HSV Color Space
    # ---------------------------------------------------------
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ---------------------------------------------------------
    # Step 2: Define Color Bounds & Create the Mask
    # ---------------------------------------------------------
    # These values target "dark golden/brown". 
    # You will likely need to tweak these exact numbers for your specific image!
    # lower_brown = np.array([0, 0, 0])   # Lower bound: Hue 10, low saturation, very dark
    # upper_brown = np.array([176, 121, 85]) # Upper bound: Hue 30, high saturation, bright
    # Mask 1: Catch the low reds/browns (0 to 10)
    lower_1 = np.array([0, 0, 0])
    upper_1 = np.array([30, 120, 129])
    mask1 = cv2.inRange(hsv_image, lower_1, upper_1)

    # Mask 2: Catch the high reds (170 to 179)
    lower_2 = np.array([166, 0, 0])
    upper_2 = np.array([179, 120, 120])
    mask2 = cv2.inRange(hsv_image, lower_2, upper_2)

    # Combine them!
    binary_mask = cv2.bitwise_or(mask1, mask2)
    
    # cv2.inRange acts as a strict filter. 
    # If a pixel falls inside those bounds, it turns WHITE. Otherwise, BLACK.
    # binary_mask = cv2.inRange(hsv_image, lower_brown, upper_brown)

    # ---------------------------------------------------------
    # Step 3: Noise Reduction (Morphology)
    # ---------------------------------------------------------
    kernel = np.ones((8, 8), np.uint8)
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

    # ---------------------------------------------------------
    # Step 4: Contour Detection and Final Drawing
    # ---------------------------------------------------------
    # Using RETR_EXTERNAL to get the outer boundaries
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    final_output = img_rgb.copy()
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filtering out tiny specks of noise

        if 350000 > area > 3000: 
            # Draw green outline
            cv2.drawContours(final_output, [contour], -1, (255, 0, 0), 3)
            
            # Put area text
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(final_output, f"{int(area)}px", (cX - 20, cY), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)

    # ---------------------------------------------------------
    # Display the Storyboard
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.canvas.manager.set_window_title('HSV Coral Segmentation')

    axes[0].imshow(img_rgb)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(cleaned_mask, cmap='gray')
    axes[1].set_title('HSV Bounded Binary Image')
    axes[1].axis('off')

    axes[2].imshow(final_output)
    axes[2].set_title('Final Contours')
    axes[2].axis('off')

    # ---------------------------------------------------------
    # Window 2: Display the Final Result
    # ---------------------------------------------------------
    # Create a new figure and axis for the final output
    fig2, ax_final = plt.subplots(figsize=(10, 8))
    fig2.canvas.manager.set_window_title('Coral Computer Vision Pipeline - Final Output')

    ax_final.imshow(final_output)
    ax_final.set_title('Final Contours')
    ax_final.axis('off')

    fig2.tight_layout()

    # ---------------------------------------------------------
    # Show all generated windows simultaneously
    # ---------------------------------------------------------

    fig.tight_layout()
    plt.show()

def process_coral_kmeans(image_path):
    # ---------------------------------------------------------
    # 1. Load the Image
    # ---------------------------------------------------------
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ---------------------------------------------------------
    # Step 1: Prepare Image for K-Means
    # ---------------------------------------------------------
    # K-Means needs a 2D array of pixels, not a 3D image grid.
    # We "flatten" the image so it's just a long list of RGB color values.
    pixel_values = img_rgb.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # ---------------------------------------------------------
    # Step 2: Run K-Means Clustering
    # ---------------------------------------------------------
    # Define criteria: Stop the algorithm after 100 iterations OR 
    # if the clusters move less than 0.2 units (accuracy).
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    K = 3
    
    # Run the algorithm
    _, labels, (centers) = cv2.kmeans(pixel_values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Reconstruct the image using ONLY those 4 colors for the storyboard
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape((img_rgb.shape))

    # ---------------------------------------------------------
    # Step 3: Build the Binary Mask (The Bridge!)
    # ---------------------------------------------------------
    # Change this number (0, 1, 2, or 3) until the mask isolates the coral!
    # Because K-Means randomly assigns these IDs, you have to test which one 
    # caught the dark brown color.
    CORAL_CLUSTER_ID = 1 

    # Reshape the labels back into the shape of the 2D image
    labels_2d = labels.reshape(img_rgb.shape[0], img_rgb.shape[1])
    
    # Create the binary stencil: 
    # If the pixel belongs to the coral cluster, make it 255 (White). Else, 0 (Black).
    binary_mask = np.uint8(labels_2d == CORAL_CLUSTER_ID) * 255

    # ---------------------------------------------------------
    # Step 3.5: Remove Massive Components (The Grid)
    # ---------------------------------------------------------
    # binary_mask is your mask from K-Means where the coral/grid are white (255)
    
    # Run connected components
    # It returns the number of blobs, the labeled image, stats (like area), and center points
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    # Create a brand new, completely black mask to draw the filtered results onto
    grid_free_mask = np.zeros_like(binary_mask)
    
    # Loop through every blob found. 
    # We start at 1 because label 0 is always the black background itself!
    for i in range(1, num_labels):
        blob_area = stats[i, cv2.CC_STAT_AREA]
        
        # Set thresholds: Bigger than dust, smaller than the giant grid
        if 2500 < blob_area < 100000:
            # If it passes the test, it's coral! Draw it as white (255) on the new mask
            grid_free_mask[labels == i] = 255
    
    # ---------------------------------------------------------
    # Step 4: Cleanup & Contours
    # ---------------------------------------------------------
    # Clean up the mask using Morphology
    kernel = np.ones((5, 5), np.uint8)
    cleaned_mask = cv2.morphologyEx(grid_free_mask, cv2.MORPH_OPEN, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

    # Find the contours on the shiny new K-Means mask
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    final_output = img_rgb.copy()
    
    # Draw contours and calculate area
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if 350000 > area > 2300:
            cv2.drawContours(final_output, [contour], -1, (0, 255, 0), 3)
            
            # Put area text
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(final_output, f"{int(area)}px", (cX - 20, cY), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)

    # ---------------------------------------------------------
    # Display the Storyboard
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.canvas.manager.set_window_title('K-Means Coral Segmentation')

    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(segmented_image)
    axes[0, 1].set_title(f'K-Means Image (K={K} Colors)')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(cleaned_mask, cmap='gray')
    axes[1, 0].set_title(f'Binary Image of Cluster {CORAL_CLUSTER_ID}')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(final_output)
    axes[1, 1].set_title('Final Area Calculation')
    axes[1, 1].axis('off')

    # ---------------------------------------------------------
    # Window 2: Display the Final Result
    # ---------------------------------------------------------
    # Create a new figure and axis for the final output
    fig2, ax_final = plt.subplots(figsize=(10, 8))
    fig2.canvas.manager.set_window_title('Coral Computer Vision Pipeline - Final Output')

    ax_final.imshow(final_output)
    ax_final.set_title('Step 4: Final Contours & Pixel Area')
    ax_final.axis('off')

    fig2.tight_layout()

    plt.tight_layout()
    plt.show()

def nothing(x):
    pass

def calibrate_hsv(image_path):
    # 1. Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return

    # Resize the image so it fits on your screen during calibration
    # Adjust these dimensions if it's still too large or too small
    image = cv2.resize(image, (800, 600))
    
    # Convert to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 2. Create a window named 'Trackbars'
    cv2.namedWindow('Trackbars', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Trackbars', 400, 300)

    # 3. Create 6 sliders (Min/Max for H, S, and V)
    # Hue goes from 0 to 179. Saturation and Value go from 0 to 255.
    cv2.createTrackbar('Hue Min', 'Trackbars', 0, 179, nothing)
    cv2.createTrackbar('Hue Max', 'Trackbars', 179, 179, nothing)
    cv2.createTrackbar('Sat Min', 'Trackbars', 0, 255, nothing)
    cv2.createTrackbar('Sat Max', 'Trackbars', 255, 255, nothing)
    cv2.createTrackbar('Val Min', 'Trackbars', 0, 255, nothing)
    cv2.createTrackbar('Val Max', 'Trackbars', 255, 255, nothing)

    print("Adjust the sliders until the coral is WHITE and everything else is BLACK.")
    print("Press the 'q' key on your keyboard to quit.")

    while True:
        # 4. Read the current position of all 6 sliders
        h_min = cv2.getTrackbarPos('Hue Min', 'Trackbars')
        h_max = cv2.getTrackbarPos('Hue Max', 'Trackbars')
        s_min = cv2.getTrackbarPos('Sat Min', 'Trackbars')
        s_max = cv2.getTrackbarPos('Sat Max', 'Trackbars')
        v_min = cv2.getTrackbarPos('Val Min', 'Trackbars')
        v_max = cv2.getTrackbarPos('Val Max', 'Trackbars')

        # 5. Create arrays for the lower and upper bounds
        lower_bound = np.array([h_min, s_min, v_min])
        upper_bound = np.array([h_max, s_max, v_max])

        # 6. Generate the mask based on the slider positions
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

        # Optional: Combine the mask with the original image to see the actual colors shining through
        result = cv2.bitwise_and(image, image, mask=mask)

        # 7. Display the original image, the black/white mask, and the color result
        cv2.imshow('Original Image', image)
        cv2.imshow('Binary Mask', mask)
        cv2.imshow('Color Result', result)

        # Wait for 1 millisecond and check if the 'q' key was pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n--- YOUR FINAL HSV VALUES ---")
            print(f"lower_bound = np.array([{h_min}, {s_min}, {v_min}])")
            print(f"upper_bound = np.array([{h_max}, {s_max}, {v_max}])")
            break

    cv2.destroyAllWindows()

def process_coral_backprojection(image_path):
    # ---------------------------------------------------------
    # 1. Load the Image
    # ---------------------------------------------------------
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return
    
    # Resize for easier viewing during the ROI selection
    # You can remove this if you want to process the full 4K resolution
    img_resized = cv2.resize(img, (1000, 750))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    hsv_image = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)

    # ---------------------------------------------------------
    # Step 1: Interactive ROI Selection (The "Training" Data)
    # ---------------------------------------------------------
    print("A window will open. Click and drag to draw a box around ONE piece of dark coral.")
    print("Make sure NOT to include the black grid or white tile in your box!")
    print("Press ENTER or SPACE when done. Press 'c' to cancel.")
    
    # Open window to let user draw a bounding box
    roi_coords = cv2.selectROI("Select Coral Sample", img_resized, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select Coral Sample")
    
    # Extract the coordinates (x, y, width, height)
    x, y, w, h = roi_coords
    
    # If the user didn't select anything, exit the script
    if w == 0 or h == 0:
        print("No sample selected. Exiting.")
        return

    # Crop the image to just the coral sample and convert to HSV
    roi_crop = img_resized[y:y+h, x:x+w]
    hsv_roi = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2HSV)

    # ---------------------------------------------------------
    # Step 2: Calculate Histogram & Backproject
    # ---------------------------------------------------------
    # Calculate the 2D color histogram of the sample (using Hue and Saturation channels)
    # Channels [0, 1] = Hue and Saturation. 
    # Bins [180, 256] = All possible values. 
    # Ranges [0, 180, 0, 256] = The min/max limits.
    roi_hist = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    
    # Normalize the histogram so values range from 0 to 255
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    
    # Apply Backprojection to the entire image using the sample's histogram
    probability_map = cv2.calcBackProject([hsv_image], [0, 1], roi_hist, [0, 180, 0, 256], 1)

    # Smooth the probability map using a circular filter (removes blocky noise)
    disc_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cv2.filter2D(probability_map, -1, disc_kernel, probability_map)

    # ---------------------------------------------------------
    # Step 3: Threshold to create Binary Mask
    # ---------------------------------------------------------
    # If a pixel has a probability > 50 of being coral, turn it white (255)
    _, binary_mask = cv2.threshold(probability_map, 50, 255, cv2.THRESH_BINARY)

    # Cleanup the mask using Morphology (to remove grid lines or sand)
    kernel = np.ones((5, 5), np.uint8)
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

    # ---------------------------------------------------------
    # Step 4: Contours and Measurement
    # ---------------------------------------------------------
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    final_output = img_rgb.copy()
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Area filter to ignore tiny specks
        if area > 100: 
            cv2.drawContours(final_output, [contour], -1, (0, 255, 0), 2)
            
            # Put area text
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(final_output, f"{int(area)}", (cX - 15, cY), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    # ---------------------------------------------------------
    # Display the Storyboard
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.canvas.manager.set_window_title('Histogram Backprojection Pipeline')

    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(probability_map, cmap='gray')
    axes[0, 1].set_title('Probability Map (White = Likely Coral)')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(cleaned_mask, cmap='gray')
    axes[1, 0].set_title('Thresholded Binary Mask')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(final_output)
    axes[1, 1].set_title('Final Coral Detection')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

def process_coral_with_texture(image_path):
    # ---------------------------------------------------------
    # 1. Load the Image
    # ---------------------------------------------------------
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return
    
    # We might want to resize if the image is massive, LBP can be slow on 4K images
    img = cv2.resize(img, (800, 600))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---------------------------------------------------------
    # Step 1: Calculate the Texture Map (LBP)
    # ---------------------------------------------------------
    # LBP settings: We look at a radius of 3 pixels around every pixel
    radius = 3
    n_points = 8 * radius
    
    # Calculate Local Binary Pattern 
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    
    # The LBP algorithm outputs tiny decimals. We need to normalize it 
    # so it scales perfectly from 0 (Smooth) to 255 (Extremely Rough)
    lbp_normalized = np.uint8((lbp / lbp.max()) * 255)

    # ---------------------------------------------------------
    # Step 2: Feature Stacking for K-Means
    # ---------------------------------------------------------
    # img_rgb has 3 channels (R, G, B). We use numpy's "depth stack" to 
    # glue the LBP map on as a 4th channel. 
    combined_features = np.dstack((img_rgb, lbp_normalized))
    
    # Flatten into a list of pixels. Now each pixel is [R, G, B, Texture]
    pixel_values = combined_features.reshape((-1, 4))
    pixel_values = np.float32(pixel_values)

    # ---------------------------------------------------------
    # Step 3: Run K-Means Clustering on the 4D Data
    # ---------------------------------------------------------
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    # We can use K=4 or K=5. Because we added texture, the algorithm has 
    # a much easier time splitting the dark colors apart.
    K = 4 
    _, labels, centers = cv2.kmeans(pixel_values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # ---------------------------------------------------------
    # Step 4: Isolate the Coral Mask
    # ---------------------------------------------------------
    # Change this ID (0, 1, 2, or 3) until it selects the rough coral!
    CORAL_CLUSTER_ID = 1
    
    labels_2d = labels.reshape(img_rgb.shape[0], img_rgb.shape[1])
    binary_mask = np.uint8(labels_2d == CORAL_CLUSTER_ID) * 255

    # Basic cleanup to remove tiny specs of noise
    kernel = np.ones((5, 5), np.uint8)
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

    # ---------------------------------------------------------
    # Step 5: Contours and Drawing
    # ---------------------------------------------------------
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_output = img_rgb.copy()
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 150: # Filter out dust
            cv2.drawContours(final_output, [contour], -1, (0, 255, 0), 2)

    # ---------------------------------------------------------
    # Display the Storyboard
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.canvas.manager.set_window_title('LBP Texture + K-Means Segmentation')

    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # Display the texture map (Bright = Rough/Bumpy, Dark = Smooth)
    axes[0, 1].imshow(lbp_normalized, cmap='magma')
    axes[0, 1].set_title('LBP Texture Map')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(cleaned_mask, cmap='gray')
    axes[1, 0].set_title(f'Binary Mask (Cluster {CORAL_CLUSTER_ID})')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(final_output)
    axes[1, 1].set_title('Final Coral Detection')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

# Replace with your image file
# process_coral_with_texture('T-101_DHEL-11__DHEL-12_20240802_BROOD.jpg')
# Run the function
# process_coral_backprojection('T-101_DHEL-11__DHEL-12_20240802_BROOD.jpg')

# Use the name of your uploaded file
# calibrate_hsv('T-101_DHEL-11__DHEL-12_20240802_BROOD.jpg')

# Replace 'coral.jpg' with your image file name
process_coral_kmeans('T-101_DHEL-11__DHEL-12_20240802_BROOD.JPG')
# process_coral_image('T-101_DHEL-11__DHEL-12_20240802_BROOD.JPG')
# process_coral_hsv('T-101_DHEL-11__DHEL-12_20240802_BROOD.JPG')







#=========================================================================================
#
#=========================================================================================






# # 1. Load the image in grayscale
# gray_image = cv2.imread('T-101_DHEL-11__DHEL-12_20240802_BROOD.JPG', cv2.IMREAD_GRAYSCALE)

# # 2. Apply Gaussian Blur to reduce noise
# # The (5, 5) is the kernel size (must be odd numbers). Higher = blurrier.
# blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# # 3. Apply Sobel Filter (Just for visualization, Canny does this internally)
# # cv2.CV_64F allows for negative numbers during calculation
# sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3) # Horizontal changes
# sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3) # Vertical changes

# # 4. Apply Canny Edge Detection
# # The numbers 50 and 150 are the lower and upper thresholds for edge linking.
# # Any gradient above 150 is definitely an edge. Any gradient below 50 is discarded.
# canny_edges = cv2.Canny(blurred_image, 50, 150)

# # Display the results
# fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# fig.canvas.manager.set_window_title('Edge Detectors and Gaussian')
# axes[0,0].imshow(blurred_image, cmap='gray')
# axes[0,0].set_title('Gaussian Blur')
# axes[0, 0].axis('off')

# axes[0,1].imshow(canny_edges, cmap='gray')
# axes[0,1].set_title('Canny Edges')
# axes[0, 1].axis('off')

# axes[1,0].imshow(sobel_y, cmap='gray') # Showing only X gradients for example
# axes[1,0].set_title('Sobel (Y-Direction)')
# axes[1, 0].axis('off')

# axes[1,1].imshow(sobel_x, cmap='gray') # Showing only X gradients for example
# axes[1,1].set_title('Sobel (X-Direction)')
# axes[1, 1].axis('off')



# plt.tight_layout()
# plt.show()