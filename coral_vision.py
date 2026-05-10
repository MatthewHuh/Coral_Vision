import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.feature import local_binary_pattern

def process_coral_image(image_path):
    # ---------------------------------------------------------
    # 1. Load the Image
    # ---------------------------------------------------------
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ---------------------------------------------------------
    # Step 1: Grayscale Conversion
    # ---------------------------------------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---------------------------------------------------------
    # Step 2: Thresholding (The Binary Mask)
    # ---------------------------------------------------------
    _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # ---------------------------------------------------------
    # Step 3: Noise Reduction
    # ---------------------------------------------------------
    kernel = np.ones((5, 5), np.uint8)
    
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

    # ---------------------------------------------------------
    # Step 4 & 5: Contour Detection and Final Measurement
    # ---------------------------------------------------------
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    final_output = img_rgb.copy()
    
    for contour in contours:
        # Calculate the area of the contour in pixels
        area = cv2.contourArea(contour)

        # Filter out contours that are too small or too large
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

    fig2, ax_final = plt.subplots(figsize=(10, 8))
    fig2.canvas.manager.set_window_title('Coral Computer Vision Pipeline - Final Output')

    ax_final.imshow(final_output)
    ax_final.set_title('Step 4: Final Contours & Pixel Area')
    ax_final.axis('off')

    fig2.tight_layout()

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
    lower_1 = np.array([0, 0, 0])
    upper_1 = np.array([30, 120, 129])
    mask1 = cv2.inRange(hsv_image, lower_1, upper_1)

    lower_2 = np.array([166, 0, 0])
    upper_2 = np.array([179, 120, 120])
    mask2 = cv2.inRange(hsv_image, lower_2, upper_2)

    binary_mask = cv2.bitwise_or(mask1, mask2)
    

    # ---------------------------------------------------------
    # Step 3: Noise Reduction (Morphology)
    # ---------------------------------------------------------
    kernel = np.ones((8, 8), np.uint8)
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

    # ---------------------------------------------------------
    # Step 4: Contour Detection and Final Drawing
    # ---------------------------------------------------------
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    final_output = img_rgb.copy()
    
    for contour in contours:
        area = cv2.contourArea(contour)
        

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
    pixel_values = img_rgb.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # ---------------------------------------------------------
    # Step 2: Run K-Means Clustering
    # ---------------------------------------------------------
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    K = 3
    
    _, labels, (centers) = cv2.kmeans(pixel_values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Reconstruct the image using ONLY those 4 colors for the storyboard
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape((img_rgb.shape))

    # ---------------------------------------------------------
    # Step 3: Build the Binary Mask 
    # ---------------------------------------------------------
    # Change this number until the mask isolates the coral
    CORAL_CLUSTER_ID = 1 

    labels_2d = labels.reshape(img_rgb.shape[0], img_rgb.shape[1])
    
    binary_mask = np.uint8(labels_2d == CORAL_CLUSTER_ID) * 255

    # ---------------------------------------------------------
    # Step 4: Remove Massive Components
    # ---------------------------------------------------------
    
    # Run connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    grid_free_mask = np.zeros_like(binary_mask)
    
    for i in range(1, num_labels):
        blob_area = stats[i, cv2.CC_STAT_AREA]
        
        if 2500 < blob_area < 100000:
            grid_free_mask[labels == i] = 255
    
    # ---------------------------------------------------------
    # Step 5: Cleanup & Contours
    # ---------------------------------------------------------
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
    roi_hist = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    
    # Normalize the histogram so values range from 0 to 255
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    
    # Apply Backprojection to the entire image using the sample's histogram
    probability_map = cv2.calcBackProject([hsv_image], [0, 1], roi_hist, [0, 180, 0, 256], 1)

    # Smooth the probability map using a circular filter 
    disc_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cv2.filter2D(probability_map, -1, disc_kernel, probability_map)

    # ---------------------------------------------------------
    # Step 3: Threshold to create Binary Mask
    # ---------------------------------------------------------
    _, binary_mask = cv2.threshold(probability_map, 50, 255, cv2.THRESH_BINARY)

    # Cleanup the mask 
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



# Uncoment the method you want to run
# process_coral_backprojection('T-101_DHEL-11__DHEL-12_20240802_BROOD.jpg')
# calibrate_hsv('T-101_DHEL-11__DHEL-12_20240802_BROOD.jpg')
process_coral_kmeans('T-101_DHEL-11__DHEL-12_20240802_BROOD.JPG')
# process_coral_image('T-101_DHEL-11__DHEL-12_20240802_BROOD.JPG')
# process_coral_hsv('T-101_DHEL-11__DHEL-12_20240802_BROOD.JPG')
