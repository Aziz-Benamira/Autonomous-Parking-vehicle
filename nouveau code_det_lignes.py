import numpy as np
import cv2
from moviepy import editor

# 1. Color filter to detect white and yellow lanes
def color_filter(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # White mask
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([255, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    # Yellow mask
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    # Combine masks
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(image, image, mask=combined_mask)

# 2. Tighter Region of Interest (trapezoid)
def region_selection(image):
    mask = np.zeros_like(image)
    rows, cols = image.shape[:2]
    bottom_left  = [cols * 0.15, rows * 0.95]
    top_left     = [cols * 0.45, rows * 0.6]
    top_right    = [cols * 0.55, rows * 0.6]
    bottom_right = [cols * 0.85, rows * 0.95]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    if len(image.shape) > 2:
        ignore_mask_color = (255,) * image.shape[2]
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    return cv2.bitwise_and(image, mask)

# 3. Hough Transform for line detection
def hough_transform(image):
    return cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=20,
                           minLineLength=20, maxLineGap=300)

# 4. Average slope and intercept with filtering
def average_slope_intercept(lines):
    left_lines, left_weights = [], []
    right_lines, right_weights = [], []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.5:
                continue  # filter out near-horizontal lines (shadows, etc.)
            intercept = y1 - slope * x1
            length = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append(length)
            else:
                right_lines.append((slope, intercept))
                right_weights.append(length)
    left_lane  = np.dot(left_weights,  left_lines) / np.sum(left_weights)  if left_weights else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if right_weights else None
    return left_lane, right_lane

# 5. Convert slope-intercept lines to pixel coordinates
def pixel_points(y1, y2, line):
    if line is None:
        return None
    slope, intercept = line
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return ((x1, int(y1)), (x2, int(y2)))

# 6. Compute lane lines
def lane_lines(image, lines):
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]
    y2 = y1 * 0.6
    return pixel_points(y1, y2, left_lane), pixel_points(y1, y2, right_lane)

# 7. Draw the detected lines
def draw_lane_lines(image, lines, color=[0, 255, 0], thickness=10):
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line, color, thickness)
    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0)

# 8. Full frame processor pipeline
def frame_processor(image):
    filtered = color_filter(image)
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    roi = region_selection(edges)
    lines = hough_transform(roi)
    if lines is None:
        return image  # No lines detected
    line_coords = lane_lines(image, lines)
    return draw_lane_lines(image, line_coords)

# 9. Apply to video
def process_video(input_path, output_path):
    input_video = editor.VideoFileClip(input_path, audio=False)
    processed = input_video.fl_image(frame_processor)
    processed.write_videofile(output_path, audio=False)

# Run on example video
process_video('input2.mp4', 'output_lines.mp4')
