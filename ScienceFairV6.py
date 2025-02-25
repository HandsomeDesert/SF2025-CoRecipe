import os
import numpy as np
from vision_agent.tools import *
from vision_agent.tools.planner_tools import judge_od_results
from typing import *
from pillow_heif import register_heif_opener
register_heif_opener()
import vision_agent as va
from vision_agent.tools import register_tool

from vision_agent.tools import load_image, owlv2_object_detection, overlay_bounding_boxes, save_image

def solve_fridge_contents(image_path: str) -> dict:
    """
    Analyze the contents of a fridge image, detect food items, and suggest recipes based on the detected ingredients.
    
    Parameters
    ----------
    image_path : str
        The path to the refrigerator image.

    Returns
    -------
    dict
        A dictionary containing:
            "available_ingredients": A dict of detected ingredients with highest confidence,
            "recipes": A list of recipe suggestions, each containing:
                - "recipe_name": Name of the recipe
                - "essential_missing": Essential ingredients that are missing
                - "recommended_missing": Recommended ingredients that are missing
                - "recommended_available": Recommended ingredients that are available
                - "optional_available": Optional ingredients that are available
                - "instructions": Preparation instructions
    """
    # 1) Load the image
    image = load_image(image_path)

    # 2) Use the owlv2_object_detection tool with a comprehensive prompt
    food_prompt = (
        "milk, yogurt, cheese, butter, cream, eggs, "
        "apple, banana, orange, grape, lemon, pineapple, strawberry, berry, "
        "carrot, broccoli, pepper, tomato, lettuce, cucumber, onion, celery, garlic, radish, "
        "chicken, turkey, ham, sausage, hot dog, fish, "
        "pasta, salad, dip, sauce, "
        "juice, water, soda"
    )
    detections = owlv2_object_detection(food_prompt, image)

    # 3) Create a dictionary of detected ingredients with their best confidence score
    ingredient_inventory = {}
    for det in detections:
        if det["score"] > 0.3:
            label = det["label"]
            ingredient_inventory[label] = max(ingredient_inventory.get(label, 0), det["score"])

    # 4) Define recipes with essential, recommended, and optional ingredients
    recipes = {
        "Fresh Fruit Salad": {
            "essential": {"apple", "banana"},
            "recommended": {"orange", "grape", "strawberry"},
            "optional": {"pineapple"},
            "instructions": "Wash and cut fruits into bite-sized pieces. Mix in a bowl."
        },
        "Vegetable Stir-Fry": {
            "essential": {"carrot", "broccoli"},
            "recommended": {"pepper", "onion"},
            "optional": {"celery", "garlic"},
            "instructions": "Cut vegetables uniformly. Stir-fry with oil and seasonings."
        },
        "Protein-Packed Breakfast": {
            "essential": {"eggs", "cheese"},
            "recommended": {"ham", "pepper"},
            "optional": {"tomato", "onion"},
            "instructions": "Scramble eggs with cheese, add protein and vegetables."
        },
        "Garden Fresh Salad": {
            "essential": {"lettuce", "tomato"},
            "recommended": {"carrot", "cucumber"},
            "optional": {"radish", "pepper"},
            "instructions": "Chop vegetables, combine in a bowl, add dressing."
        }
    }

    # Convert to a set for easier membership checks
    available_ingredients_set = set(ingredient_inventory.keys())
    
    # 5) Check recipes and build a result
    recipe_suggestions = []
    for recipe_name, recipe_info in recipes.items():
        essential_missing = recipe_info["essential"] - available_ingredients_set
        recommended_missing = recipe_info["recommended"] - available_ingredients_set
        recommended_available = recipe_info["recommended"].intersection(available_ingredients_set)
        optional_available = recipe_info["optional"].intersection(available_ingredients_set)
        
        recipe_suggestions.append({
            "recipe_name": recipe_name,
            "essential_missing": sorted(list(essential_missing)),
            "recommended_missing": sorted(list(recommended_missing)),
            "recommended_available": sorted(list(recommended_available)),
            "optional_available": sorted(list(optional_available)),
            "instructions": recipe_info["instructions"]
        })
    
    # 6) Visualize bounding boxes on the image
    result_image = overlay_bounding_boxes(image, detections)
    save_image(result_image, "detected_fridge_food.jpg")
    
    # 7) Return the final structured results
    return {
        "available_ingredients": {
            ing: round(score, 2) for ing, score in sorted(
                ingredient_inventory.items(), key=lambda x: x[1], reverse=True
            )
        },
        "recipes": recipe_suggestions
    }
