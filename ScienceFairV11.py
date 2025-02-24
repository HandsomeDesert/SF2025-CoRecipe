"""
This is my Science Fair project 2025. You will need to download the modules or use the visionagent by landingai website.
"""

import os
import numpy as np
from vision_agent.tools import *
from vision_agent.tools.planner_tools import judge_od_results
from typing import *
from pillow_heif import register_heif_opener
register_heif_opener()
import vision_agent as va
from vision_agent.tools import register_tool

from typing import List, Dict
from vision_agent.tools import load_image, qwen2_vl_images_vqa

def analyze_fridge_and_suggest_recipes(image_path: str) -> str:
    """
    This function takes the path to a fridge image, identifies the
    food items available, and suggests suitable recipes along with
    their instructions and nutritional information.

    Parameters
    ----------
    image_path : str
        A string representing the file path or URL to the fridge image.

    Returns
    -------
    str
        A formatted string that includes the identified fridge items,
        the recipes that can be made, their respective steps, prep time,
        servings, nutritional information, and a summary of all recipes
        possible with the identified ingredients.
    """

    # Load the fridge image
    fridge_image = load_image(image_path)

    # Obtain the list of items in the fridge using the visual QA tool
    # (Tool obtained from get_tool_for_task in planning.)
    fridge_inventory = qwen2_vl_images_vqa(
        "List all the food items you can see in this fridge, categorized by type (fruits & vegetables, meats, dairy). Be specific and detailed. Don't say any other words. Don't repeat any ingredients",
        [fridge_image]
    )

    # Define recipes (with required and optional ingredients)
    recipes = [
        {
            "name": "Fresh Fruit Medley",
            "required_ingredients": ["apple", "grape"],
            "optional_ingredients": ["pineapple", "orange", "pear"],
            "prep_time": "10 minutes",
            "servings": 4,
            "nutrition": "Rich in vitamins C and fiber",
            "instructions": (
                "1. Wash all fruits thoroughly\n"
                "2. Cut apples and pears into bite-sized chunks\n"
                "3. Halve the grapes\n"
                "4. If using pineapple, cut into small chunks\n"
                "5. Mix all fruits in a large bowl\n"
                "6. Optional: Add a squeeze of orange juice to prevent browning"
            ),
        },
        {
            "name": "Colorful Garden Salad",
            "required_ingredients": ["lettuce", "tomato", "carrot", "cucumber"],
            "optional_ingredients": ["bell pepper", "radish"],
            "prep_time": "15 minutes",
            "servings": 4,
            "nutrition": "High in vitamins A, C, and K",
            "instructions": (
                "1. Wash and dry all vegetables\n"
                "2. Tear lettuce into bite-sized pieces\n"
                "3. Slice tomatoes, cucumbers, and radishes\n"
                "4. Grate or julienne carrots\n"
                "5. If using, slice bell peppers\n"
                "6. Combine all ingredients\n"
                "7. Serve with your favorite dressing"
            ),
        },
        {
            "name": "Protein-Packed Deli Plate",
            "required_ingredients": ["ham", "cheese", "egg"],
            "optional_ingredients": ["cucumber", "tomato"],
            "prep_time": "10 minutes",
            "servings": 2,
            "nutrition": "High in protein and calcium",
            "instructions": (
                "1. Arrange sliced ham and cheese on a plate\n"
                "2. Hard boil eggs, cool, and slice\n"
                "3. If using, slice cucumber and tomato for garnish\n"
                "4. Serve with whole grain crackers or bread"
            ),
        },
        {
            "name": "Broccoli and Cheese Bake",
            "required_ingredients": ["broccoli", "cheese"],
            "optional_ingredients": ["carrot"],
            "prep_time": "25 minutes",
            "servings": 4,
            "nutrition": "Good source of calcium and vitamin C",
            "instructions": (
                "1. Cut broccoli into florets\n"
                "2. Steam broccoli until tender-crisp\n"
                "3. Grate cheese\n"
                "4. If using, slice carrots thinly\n"
                "5. Layer vegetables in a baking dish\n"
                "6. Top with cheese and bake until melted"
            ),
        },
        {
            "name": "Sausage and Pepper Medley",
            "required_ingredients": ["sausage", "bell pepper"],
            "optional_ingredients": ["onion", "tomato"],
            "prep_time": "20 minutes",
            "servings": 3,
            "nutrition": "High in protein and vitamin C",
            "instructions": (
                "1. Slice sausages into rounds\n"
                "2. Cut bell peppers into strips\n"
                "3. If using, slice tomatoes\n"
                "4. Sauté sausages until browned\n"
                "5. Add peppers and cook until tender-crisp\n"
                "6. Add tomatoes if using and heat through"
            ),
        }
    ]

    # Helper function to check if a recipe can be made with the identified inventory
    def check_recipe_availability(inventory_text: str, recipe: Dict) -> Dict:
        inventory_lower = inventory_text.lower()
        missing_required = []
        for ingredient in recipe["required_ingredients"]:
            if ingredient.lower() not in inventory_lower:
                missing_required.append(ingredient)
        available_optional = []
        for ingredient in recipe["optional_ingredients"]:
            if ingredient.lower() in inventory_lower:
                available_optional.append(ingredient)
        return {
            "can_make": len(missing_required) == 0,
            "missing_required": missing_required,
            "available_optional": available_optional
        }

    # Prepare output
    output_lines = []
    output_lines.append("🏪 YOUR FRIDGE INVENTORY:\n" + "=" * 50)
    output_lines.append(fridge_inventory)

    # Check each recipe for availability
    makeable_recipes = []
    output_lines.append("\n👩‍🍳 RECIPE SUGGESTIONS\n" + "=" * 50)

    for recipe in recipes:
        availability = check_recipe_availability(fridge_inventory, recipe)
        if availability["can_make"]:
            makeable_recipes.append(recipe["name"])
            output_lines.append(f"\n📝 {recipe['name']}:")
            output_lines.append(f"⏱️ Prep Time: {recipe['prep_time']}")
            output_lines.append(f"🍽️ Servings: {recipe['servings']}")
            output_lines.append(f"💪 Nutrition: {recipe['nutrition']}")
            if availability["available_optional"]:
                output_lines.append(
                    f"✨ Optional ingredients available: {', '.join(availability['available_optional'])}"
                )
            output_lines.append("\n📋 Instructions:")
            output_lines.append(recipe["instructions"])
            output_lines.append("-" * 50)

    # Final summary
    output_lines.append("\n📊 SUMMARY\n" + "=" * 50)
    output_lines.append(f"Based on your fridge contents, you can make {len(makeable_recipes)} recipes:")
    for i, recipe_name in enumerate(makeable_recipes, 1):
        output_lines.append(f"{i}. {recipe_name}")

    # Return the entire output as a single string
    return "\n".join(output_lines)
