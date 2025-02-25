import os
import numpy as np
from vision_agent.tools import *
from vision_agent.tools.planner_tools import judge_od_results
from typing import *
from pillow_heif import register_heif_opener
register_heif_opener()
import vision_agent as va
from vision_agent.tools import register_tool

# The main code doesn't need changes since the error is in the test
def solve_fridge_image_task(image_path: str) -> str:
    """
    This function loads a fridge image, detects common foods using a vision-based
    object detection model, categorizes them, generates a meal plan, visualizes the
    bounding boxes on the image, and finally returns the meal plan text.

    Parameters:
    -----------
    image_path : str
        The path to the fridge image (can be a local file path or a URL).

    Returns:
    --------
    str
        A detailed meal plan text that includes suggested recipes, storage tips,
        and ways to use the detected ingredients.

    Usage:
    ------
        plan_text = solve_fridge_image_task('my_fridge.jpg')
        print(plan_text)
    """
    from vision_agent.tools import load_image, owlv2_object_detection, qwen2_vl_images_vqa
    from vision_agent.tools import overlay_bounding_boxes, save_image

    # 1) Load the image
    image = load_image(image_path)

    # 2) Detect foods using an object detection model specialized in known fridge items
    foods_prompt = (
        "apple, orange, pineapple, strawberry, pepper, broccoli, tomato, "
        "lettuce, carrot, cheese, sausage"
    )
    detections = owlv2_object_detection(foods_prompt, image)
    filtered_detections = [det for det in detections if det.get("score", 0) > 0.3]

    # 3) Categorize detected foods (simple approach by label)
    categories = {
        "fruits": {"apple", "orange", "pineapple", "strawberry"},
        "vegetables": {"pepper", "broccoli", "tomato", "lettuce", "carrot"},
        "proteins": {"cheese", "sausage"},
    }
    categorized_foods = {"fruits": [], "vegetables": [], "proteins": []}
    for det in filtered_detections:
        label = det.get("label", "").lower()
        for cat, items in categories.items():
            if label in items:
                categorized_foods[cat].append(label)

    # 4) Generate meal plan or recipes using language-based vision tool
    # Build a prompt describing the detected items
    fruits_found = list(set(categorized_foods["fruits"]))
    vegetables_found = list(set(categorized_foods["vegetables"]))
    proteins_found = list(set(categorized_foods["proteins"]))

    prompt = (
        f"Detected ingredients:\n"
        f"Fruits: {fruits_found}\n"
        f"Vegetables: {vegetables_found}\n"
        f"Proteins: {proteins_found}\n"
        "Please provide a final comprehensive meal plan with breakfast, lunch, "
        "and dinner recipe ideas, plus tips on ingredient usage, storage, and "
        "quick combinations."
    )

    meal_plan_text = qwen2_vl_images_vqa(prompt, [image])

    # 5) Overlay bounding boxes for visualization
    image_with_boxes = overlay_bounding_boxes(image, filtered_detections)

    # 6) Save the visualization
    save_image(image_with_boxes, "detected_items.jpg")

    # 7) Return the meal plan text
    return meal_plan_text
