import cv2
import numpy as np
import os

model_name = input("Enter model name (e.g., sac, td3): ").strip().lower()
run_number = input("Enter run number (e.g., 1 for r1, 2 for r2): ").strip().lower()

if model_name == 's':
    model_name = 'sac'
elif model_name == 't':
    model_name = 'td3'

cw_filename = f"{model_name}_r{run_number}_cw.png"
ccw_filename = f"{model_name}_r{run_number}_ccw.png"
output_filename = f"{model_name}_r{run_number}_combined.png"

if not os.path.exists(cw_filename) or not os.path.exists(ccw_filename):
    print(f"\nError: Could not find files. Looked for: '{cw_filename}' and '{ccw_filename}'")
    exit()

img_cw = cv2.imread(cw_filename)   
img_ccw = cv2.imread(ccw_filename) 

h, w = img_cw.shape[:2]

x1, y1 = int(w * 0.32), int(h * 0.34)
x2, y2 = int(w * 0.68), int(h * 0.66)

inner_mask = np.zeros((h, w), dtype="uint8")

radius = 25 

cv2.circle(inner_mask, (x1 + radius, y1 + radius), radius, 255, thickness=cv2.FILLED) # Top-Left
cv2.circle(inner_mask, (x2 - radius, y1 + radius), radius, 255, thickness=cv2.FILLED) # Top-Right
cv2.circle(inner_mask, (x1 + radius, y2 - radius), radius, 255, thickness=cv2.FILLED) # Bottom-Left
cv2.circle(inner_mask, (x2 - radius, y2 - radius), radius, 255, thickness=cv2.FILLED) # Bottom-Right

cv2.rectangle(inner_mask, (x1 + radius, y1), (x2 - radius, y2), 255, thickness=cv2.FILLED) # Horizontal body
cv2.rectangle(inner_mask, (x1, y1 + radius), (x2, y2 - radius), 255, thickness=cv2.FILLED) # Vertical body

outer_mask = cv2.bitwise_not(inner_mask)


debug_img = img_cw.copy()
cv2.circle(debug_img, (x1 + radius, y1 + radius), radius, (255, 255, 0), thickness=2)
cv2.circle(debug_img, (x2 - radius, y1 + radius), radius, (255, 255, 0), thickness=2)
cv2.circle(debug_img, (x1 + radius, y2 - radius), radius, (255, 255, 0), thickness=2)
cv2.circle(debug_img, (x2 - radius, y2 - radius), radius, (255, 255, 0), thickness=2)

cv2.line(debug_img, (x1 + radius, y1), (x2 - radius, y1), (255, 255, 0), thickness=2) # Top
cv2.line(debug_img, (x1 + radius, y2), (x2 - radius, y2), (255, 255, 0), thickness=2) # Bottom
cv2.line(debug_img, (x1, y1 + radius), (x1, y2 - radius), (255, 255, 0), thickness=2) # Left
cv2.line(debug_img, (x2, y1 + radius), (x2, y2 - radius), (255, 255, 0), thickness=2) # Right

cv2.imwrite("debug_mask.png", debug_img)
print("-> Saved visual symmetric rounded-rectangle reference to 'debug_mask.png'")

inner_content = cv2.bitwise_and(img_cw, img_cw, mask=inner_mask)
outer_content = cv2.bitwise_and(img_ccw, img_ccw, mask=outer_mask)
final_composite = cv2.add(inner_content, outer_content)

cv2.imwrite(output_filename, final_composite)
print(f"Success! Symmetric rectangle-combined loop saved to: {output_filename}")