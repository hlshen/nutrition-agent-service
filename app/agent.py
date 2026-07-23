# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import os
import logging
import json
import re
import datetime
from typing import List, Dict, Any, Literal, Optional

from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import Gemini
from google.genai import types
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.preload_memory_tool import PreloadMemoryTool
from google.adk.tools import ToolContext
from pydantic import BaseModel, Field

# =====================================================================
# 0. EXPLICIT SCHEMAS FOR TOOLS AND AGENTS (Pydantic Models)
# =====================================================================

class UserMacros(BaseModel):
    fat_g: float = Field(description="Target daily fat intake in grams.")
    protein_g: float = Field(description="Target daily protein intake in grams.")
    carbs_g: float = Field(description="Target daily carbohydrates intake in grams.")

class UserProfile(BaseModel):
    user_id: str = Field(description="The unique system identifier for the customer.")
    calories: float = Field(description="Target daily calorie ceiling.")
    macros: UserMacros = Field(description="Calculated target macronutrient weights in grams.")
    allergies: List[str] = Field(description="List of ingredients/foods to strictly avoid due to allergies.")
    dislikes: List[str] = Field(description="List of foods the user dislikes.")
    diet_tag: str = Field(description="The dietary tag associated with the user, e.g., 'Keto', 'Standard'.")

class MacroSplits(BaseModel):
    fat_percent: float = Field(description="Target percentage split of daily calories from fat, between 0.0 and 1.0.")
    protein_percent: float = Field(description="Target percentage split of daily calories from protein, between 0.0 and 1.0.")
    carb_percent: float = Field(description="Target percentage split of daily calories from carbohydrates, between 0.0 and 1.0.")

class UpdateProfileData(BaseModel):
    calories: Optional[float] = Field(None, description="Target daily calorie ceiling.")
    macro_splits: Optional[MacroSplits] = Field(None, description="Target macro percentages for calorie distributions.")
    allergies: Optional[List[str]] = Field(None, description="Updated list of allergens to avoid.")
    dislikes: Optional[List[str]] = Field(None, description="Updated list of disliked foods to avoid.")
    diet_tag: Optional[str] = Field(None, description="Updated dietary tag (e.g., 'Keto', 'Vegan').")

class UpdateProfileResponse(BaseModel):
    success: bool = Field(description="True if the database transaction completed successfully.")
    updated_profile: Optional[UserProfile] = Field(None, description="The newly updated complete profile state.")
    error: Optional[str] = Field(None, description="Error message if the update operation failed.")
    recovery_instruction: Optional[str] = Field(None, description="Step-by-step guidance for recovering from the failure.")

class PantryItem(BaseModel):
    item: str = Field(description="The name of the ingredient or food item.")
    quantity_g: float = Field(description="The remaining weight of the item in grams.")
    expiry_days: int = Field(description="The estimated number of days before the item expires.")

class PantrySuppliesResponse(BaseModel):
    items: List[PantryItem] = Field(description="A list of active pantry supplies and ingredients currently in stock.")
    error: Optional[str] = Field(None, description="Error message if the query failed.")
    recovery_instruction: Optional[str] = Field(None, description="Step-by-step guidance for recovering from the failure.")

class UpdatePantryResponse(BaseModel):
    success: bool = Field(description="True if the database write transaction completed successfully.")
    updated_items: List[PantryItem] = Field(description="The updated full pantry supplies list currently in stock.")
    error: Optional[str] = Field(None, description="Error message if the update failed.")
    recovery_instruction: Optional[str] = Field(None, description="Step-by-step guidance for recovering from the failure.")

class RecipeIngredient(BaseModel):
    name: str = Field(description="Name of the ingredient.")
    qty_g: float = Field(description="Quantity required in grams.")

class Recipe(BaseModel):
    recipe_name: str = Field(description="Name of the culinary recipe.")
    macros: UserMacros = Field(description="Estimated macronutrient distribution (fat, protein, carb).")
    calories: float = Field(description="Total calories in the meal.")
    ingredients: List[RecipeIngredient] = Field(description="List of required ingredients and quantities.")
    instructions: str = Field(description="Detailed cooking instructions.")

class FetchRecipesResponse(BaseModel):
    recipes: List[Recipe] = Field(description="List of recipes that meet target limits and leverage available stocks.")
    error: Optional[str] = Field(None, description="Error message if no recipes were matched or found.")
    recovery_instruction: Optional[str] = Field(None, description="Step-by-step guidance for recovering from the failure.")


# =====================================================================
# AGENT TASK SCHEMAS (For Explicit Multi-Agent Typed Communication)
# =====================================================================

class DietPreferencesInput(BaseModel):
    user_id: str = Field(description="The unique system identifier for the customer.")
    instruction: str = Field(description="The user's query or instruction regarding their preferences or targets.")

class DietPreferencesOutput(BaseModel):
    user_id: str = Field(description="The unique system identifier for the customer.")
    profile: Optional[UserProfile] = Field(None, description="The user's active nutritional profile guidelines.")
    status_summary: str = Field(description="A clear natural language summary of the profile updates or status.")

class PantrySupplyInput(BaseModel):
    user_id: str = Field(description="The unique system identifier for the customer.")
    instruction: str = Field(description="The user's instruction regarding their pantry supplies updates or queries.")

class PantrySupplyOutput(BaseModel):
    user_id: str = Field(description="The unique system identifier for the customer.")
    items: List[PantryItem] = Field(description="The full list of active pantry supplies items.")
    status_summary: str = Field(description="A clear natural language summary of the pantry stock status or updates.")

class MealPlannerInput(BaseModel):
    user_id: str = Field(description="The unique system identifier for the customer.")
    target_calories: int = Field(description="The target calorie limit per meal.")
    exclusions: List[str] = Field(description="Ingredients to avoid.")
    pantry_items: List[PantryItem] = Field(description="List of active pantry items currently in stock.")

class MealPlannerOutput(BaseModel):
    recipes: List[Recipe] = Field(description="List of custom recipes matching the diet constraints and stock ingredients.")
    shopping_list: List[str] = Field(description="List of missing ingredients the user needs to buy.")
    weekly_meal_plan: str = Field(description="A beautifully formatted markdown meal plan and grocery list.")

# Setup PII Scrubbing and Structured Logging
from google.cloud import dlp_v2

EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_REGEX = re.compile(r"\b(?:\+?(\d{1,3}))?[-. (]*(\d{3})[-. )]*(\d{3})[-. ]*(\d{4})\b")

_dlp_client: dlp_v2.DlpServiceClient | None = None

def _get_dlp_client() -> dlp_v2.DlpServiceClient | None:
    global _dlp_client
    if _dlp_client is None:
        try:
            _dlp_client = dlp_v2.DlpServiceClient()
        except Exception:
            pass
    return _dlp_client

def scrub_pii(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    
    # Local Regex Redaction (First line of defense / offline fallback)
    text = EMAIL_REGEX.sub("[REDACTED_EMAIL]", text)
    text = PHONE_REGEX.sub("[REDACTED_PHONE]", text)
    
    # Enhanced Cloud DLP Redaction
    client = _get_dlp_client()
    if client is None:
        return text
        
    try:
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "ai5days-503118")
        parent = f"projects/{project_id}"
        
        info_types = [
            {"name": "EMAIL_ADDRESS"},
            {"name": "PHONE_NUMBER"},
            {"name": "PERSON_NAME"},
            {"name": "US_SOCIAL_SECURITY_NUMBER"},
            {"name": "IP_ADDRESS"},
        ]
        
        inspect_config = {
            "info_types": info_types,
            "min_likelihood": dlp_v2.Likelihood.LIKELIHOOD_UNSPECIFIED,
        }
        
        deidentify_config = {
            "info_type_transformations": {
                "transformations": [
                    {
                        "primitive_transformation": {
                            "replace_with_info_type_config": {}
                        }
                    }
                ]
            }
        }
        
        response = client.deidentify_content(
            request={
                "parent": parent,
                "deidentify_config": deidentify_config,
                "inspect_config": inspect_config,
                "item": {"value": text},
            }
        )
        return response.item.value
    except Exception:
        return text

class StructuredJSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": scrub_pii(record.getMessage()),
        }
        # Safely extract extra fields if present
        for field in ("user_id", "trace_id", "session_id", "agent_name", "event"):
            if hasattr(record, field):
                val = getattr(record, field)
                if val is not None:
                    log_data[field] = scrub_pii(str(val)) if field == "user_id" else val
        
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)

logger = logging.getLogger("nutrition_agent")
logger.setLevel(logging.INFO)
logger.handlers.clear()

handler = logging.StreamHandler()
handler.setFormatter(StructuredJSONFormatter())
logger.addHandler(handler)

# Initialize Firestore Client
from google.cloud import firestore
db = firestore.Client()
logger.info("Firestore client initialized successfully.")

# =====================================================================
# 1. TOOL IMPLEMENTATIONS
# =====================================================================

def get_user_profile(user_id: str) -> Dict[str, Any]:
    """Retrieves target macros, caloric limits, dislikes, and allergies from Firestore.
    
    Args:
        user_id: The unique system identifier for the customer.
        
    Returns:
        A dictionary conforming to the UserProfile schema, or a structured error response.
    """
    logger.info(f"Retrieving user profile for user: {user_id}")
    try:
        if not user_id or not isinstance(user_id, str):
            raise ValueError("The 'user_id' parameter must be a non-empty string.")
            
        doc_ref = db.collection("user_profiles").document(user_id)
        doc = doc_ref.get()
        if doc.exists:
            data = doc.to_dict()
            # Normalize and validate using UserProfile model
            profile = UserProfile(
                user_id=user_id,
                calories=float(data.get("calories", 2000.0)),
                macros=UserMacros(
                    fat_g=float(data.get("macros", {}).get("fat_g", 0.0)),
                    protein_g=float(data.get("macros", {}).get("protein_g", 0.0)),
                    carbs_g=float(data.get("macros", {}).get("carbs_g", 0.0))
                ),
                allergies=list(data.get("allergies", [])),
                dislikes=list(data.get("dislikes", [])),
                diet_tag=str(data.get("diet_tag", "Standard"))
            )
            return profile.model_dump()
        else:
            # Profile doesn't exist yet, return a structured recovery instruction
            return {
                "status": "error",
                "error_code": "PROFILE_NOT_FOUND",
                "message": f"No dietary profile found for user_id '{user_id}'.",
                "recovery_instruction": "The profile does not exist yet. Please ask the user for their caloric and macronutrient targets and food restrictions, and then call 'update_user_profile' to create it."
            }
    except Exception as e:
        logger.error(f"Error retrieving user profile for user {user_id}: {e}")
        return {
            "status": "error",
            "error_code": "RETRIEVAL_ERROR",
            "message": str(e),
            "recovery_instruction": "Please verify Firestore connection and that user_id is correct, then retry the request."
        }

def update_user_profile(user_id: str, profile_data: Dict[str, Any], tool_context: ToolContext = None) -> Dict[str, Any]:
    """Updates the user's dietary preferences and target metrics in Firestore.
    
    If calorie and macro percentage split targets are provided, the corresponding
    grams are calculated as:
    - Fat: (calories * fat_percent) / 9
    - Protein: (calories * protein_percent) / 4
    - Carbohydrates: (calories * carb_percent) / 4
    
    Args:
        user_id: The unique system identifier for the customer.
        profile_data: Key-value updates conforming to the UpdateProfileData schema.
        tool_context: Optional context for human-in-the-loop validation and approvals.
        
    Returns:
        A dictionary conforming to the UpdateProfileResponse schema.
    """
    try:
        if not user_id:
            raise ValueError("The 'user_id' parameter must be a non-empty string.")
        
        # Human-in-the-loop confirmation
        if tool_context is not None:
            if not tool_context.tool_confirmation or not tool_context.tool_confirmation.confirmed:
                tool_context.request_confirmation(
                    hint=f"Please confirm updating your dietary profile guidelines with: {profile_data}"
                )
                return {
                    "success": False,
                    "error": "CONFIRMATION_REQUIRED",
                    "recovery_instruction": "Please ask the user to explicitly confirm the profile update."
                }

        # Validate input using UpdateProfileData model
        validated_input = UpdateProfileData(**profile_data)
        logger.info(f"Updating user profile for user: {user_id} with data: {validated_input.model_dump()}")
        
        # Extract current profile or start fresh
        current_profile_dict = get_user_profile(user_id)
        if current_profile_dict.get("status") == "error":
            current_data = {
                "user_id": user_id,
                "calories": 2000.0,
                "macros": {"fat_g": 0.0, "protein_g": 0.0, "carbs_g": 0.0},
                "allergies": [],
                "dislikes": [],
                "diet_tag": "Standard"
            }
        else:
            current_data = current_profile_dict

        # Quantitatively validate the macro split percentages if provided
        if validated_input.macro_splits is not None:
            splits = validated_input.macro_splits
            total_split = splits.fat_percent + splits.protein_percent + splits.carb_percent
            if abs(total_split - 1.0) > 1e-4 and abs(total_split - 100.0) > 1e-2:
                raise ValueError(
                    f"Macro percentage splits must sum to exactly 100% (or 1.0). Total provided split sum: {total_split}."
                )
            
            # Standardize percentages to decimals if specified in 0-100 scale
            fat_pct = splits.fat_percent if splits.fat_percent < 1.0 else splits.fat_percent / 100.0
            prot_pct = splits.protein_percent if splits.protein_percent < 1.0 else splits.protein_percent / 100.0
            carb_pct = splits.carb_percent if splits.carb_percent < 1.0 else splits.carb_percent / 100.0
            
            cals = validated_input.calories if validated_input.calories is not None else current_data.get("calories", 2000.0)
            
            # Physical validation of calorie limits
            if cals < 500 or cals > 10000:
                raise ValueError(f"Daily calorie limit of {cals} kcal is outside the physically reasonable range (500 to 10000 kcal).")
            
            # Calorie-to-gram conversion math
            fat_g = round((cals * fat_pct) / 9, 1)
            protein_g = round((cals * prot_pct) / 4, 1)
            carbs_g = round((cals * carb_pct) / 4, 1)
            
            current_data["calories"] = float(cals)
            current_data["macros"] = {
                "fat_g": fat_g,
                "protein_g": protein_g,
                "carbs_g": carbs_g
            }
        elif validated_input.calories is not None:
            cals = validated_input.calories
            if cals < 500 or cals > 10000:
                raise ValueError(f"Daily calorie limit of {cals} kcal is outside the physically reasonable range (500 to 10000 kcal).")
            current_data["calories"] = float(cals)

        if validated_input.allergies is not None:
            current_data["allergies"] = validated_input.allergies
        if validated_input.dislikes is not None:
            current_data["dislikes"] = validated_input.dislikes
        if validated_input.diet_tag is not None:
            current_data["diet_tag"] = validated_input.diet_tag

        # Validate full profile structure before storing
        profile_model = UserProfile(
            user_id=user_id,
            calories=float(current_data["calories"]),
            macros=UserMacros(
                fat_g=float(current_data["macros"]["fat_g"]),
                protein_g=float(current_data["macros"]["protein_g"]),
                carbs_g=float(current_data["macros"]["carbs_g"])
            ),
            allergies=current_data["allergies"],
            dislikes=current_data["dislikes"],
            diet_tag=current_data["diet_tag"]
        )
        
        # Write strictly to Firestore
        db.collection("user_profiles").document(user_id).set(profile_model.model_dump())
        
        return UpdateProfileResponse(
            success=True,
            updated_profile=profile_model
        ).model_dump()
        
    except Exception as e:
        logger.error(f"Error updating user profile: {e}")
        recovery_instruction = (
            "Ensure all values in profile_data are correct and of the appropriate type. "
            "If macro_splits are provided, double-check that fat_percent + protein_percent + carb_percent == 1.0 (or 100%)."
        )
        if "Macro percentage" in str(e):
            recovery_instruction = "The percentages must sum up to exactly 1.0 (e.g. 0.7, 0.25, 0.05). Please ask the user to clarify if they sum to more or less."
        elif "calorie" in str(e) or "calories" in str(e):
            recovery_instruction = "The target calorie limit is out of reasonable range (500 to 10000 kcal). Please ask the user to adjust."
            
        return UpdateProfileResponse(
            success=False,
            error=str(e),
            recovery_instruction=recovery_instruction
        ).model_dump()

def query_pantry_supplies(user_id: str) -> Dict[str, Any]:
    """Retrieves a list of current ingredients available in the user's pantry.
    
    Args:
        user_id: The unique system identifier for the customer.
        
    Returns:
        A dictionary conforming to the PantrySuppliesResponse schema.
    """
    logger.info(f"Retrieving pantry supplies for user: {user_id}")
    try:
        if not user_id:
            raise ValueError("The 'user_id' parameter must be a non-empty string.")
            
        doc_ref = db.collection("pantry_supplies").document(user_id)
        doc = doc_ref.get()
        items_list = []
        if doc.exists:
            data = doc.to_dict()
            raw_items = data.get("items", [])
            for x in raw_items:
                items_list.append(PantryItem(
                    item=str(x.get("item", "")),
                    quantity_g=float(x.get("quantity_g", 0.0)),
                    expiry_days=int(x.get("expiry_days", 7))
                ))
        
        return PantrySuppliesResponse(
            items=items_list
        ).model_dump()
        
    except Exception as e:
        logger.error(f"Error querying pantry supplies: {e}")
        return PantrySuppliesResponse(
            items=[],
            error=str(e),
            recovery_instruction="Please check your Firestore database access or verify if the user's pantry supplies document is structured correctly."
        ).model_dump()

def update_pantry_supplies(user_id: str, items: List[Dict[str, Any]], operation: str = "upsert", tool_context: ToolContext = None) -> Dict[str, Any]:
    """Updates pantry inventory by adding purchased items or subtracting consumed items.
    
    Args:
        user_id: The unique system identifier for the customer.
        items: List of dictionary records conforming to the PantryItem schema.
        operation: 'upsert' to add/increment stock, 'consume' or 'delete' to subtract or remove.
        tool_context: Optional context for human-in-the-loop validation and approvals.
        
    Returns:
        A dictionary conforming to the UpdatePantryResponse schema.
    """
    try:
        if not user_id:
            raise ValueError("The 'user_id' parameter must be a non-empty string.")
        if not items:
            raise ValueError("The 'items' list cannot be empty.")
            
        if tool_context is not None:
            if not tool_context.tool_confirmation or not tool_context.tool_confirmation.confirmed:
                tool_context.request_confirmation(
                    hint=f"Please confirm updating your pantry supplies: {operation}ing items: {items}"
                )
                return {
                    "success": False,
                    "error": "CONFIRMATION_REQUIRED",
                    "recovery_instruction": "Please ask the user to explicitly confirm the pantry inventory update."
                }

        logger.info(f"Updating pantry supplies for user: {user_id} with items: {items}, operation: {operation}")
        
        # Validate input items using PantryItem schema
        input_items = [PantryItem(**x) for x in items]
        
        # Query existing supplies
        pantry_res = query_pantry_supplies(user_id)
        if pantry_res.get("error") is not None:
            raise RuntimeError(f"Could not retrieve existing pantry: {pantry_res.get('error')}")
            
        current_items = pantry_res.get("items", [])
        pantry_dict = {x["item"].lower(): PantryItem(**x) for x in current_items}
        
        for update_item in input_items:
            name = update_item.item.lower()
            qty = update_item.quantity_g
            
            if operation == "upsert":
                if name in pantry_dict:
                    pantry_dict[name].quantity_g += qty
                else:
                    pantry_dict[name] = PantryItem(
                        item=update_item.item,
                        quantity_g=qty,
                        expiry_days=update_item.expiry_days
                    )
            elif operation in ("consume", "delete"):
                if name in pantry_dict:
                    current_qty = pantry_dict[name].quantity_g
                    if current_qty <= qty or qty == 0:
                        del pantry_dict[name]
                    else:
                        pantry_dict[name].quantity_g = current_qty - qty
                        
        updated_list = list(pantry_dict.values())
        
        # Store strictly to Firestore
        db.collection("pantry_supplies").document(user_id).set({"items": [x.model_dump() for x in updated_list]})
        
        return UpdatePantryResponse(
            success=True,
            updated_items=updated_list
        ).model_dump()
        
    except Exception as e:
        logger.error(f"Error updating pantry supplies: {e}")
        return UpdatePantryResponse(
            success=False,
            updated_items=[],
            error=str(e),
            recovery_instruction="Check that all items contain an 'item' name and a positive 'quantity_g'. Verify that the operation is either 'upsert', 'consume', or 'delete'."
        ).model_dump()

def fetch_recipes(target_calories: int, exclusions: List[str], focus_ingredients: List[str]) -> Dict[str, Any]:
    """Queries a vectorized catalog for recipes using in-stock items and avoiding exclusions.
    
    Args:
        target_calories: Target per-meal caloric ceiling.
        exclusions: List of ingredients to strictly avoid.
        focus_ingredients: Stock ingredients to prioritize using first.
        
    Returns:
        A dictionary conforming to the FetchRecipesResponse schema.
    """
    logger.info(f"Fetching recipes matching target_calories: {target_calories}, exclusions: {exclusions}, focus_ingredients: {focus_ingredients}")
    try:
        if target_calories <= 0:
            raise ValueError("The 'target_calories' parameter must be a positive integer.")
            
        # Simple rule-based mock matching database records for testing and correctness
        all_recipes = [
            {
                "recipe_name": "Garlic Chicken with Spinach (Keto)",
                "macros": {"protein_g": 48, "carbs_g": 5, "fat_g": 25},
                "calories": 437,
                "ingredients": [
                    {"name": "chicken breast", "qty_g": 150},
                    {"name": "spinach", "qty_g": 50},
                    {"name": "olive oil", "qty_g": 20},
                    {"name": "garlic", "qty_g": 10}
                ],
                "instructions": "Sear chicken breast in olive oil with minced garlic. Toss in fresh spinach until wilted. Season with salt and pepper."
            },
            {
                "recipe_name": "Creamy Mushroom Chicken",
                "macros": {"protein_g": 42, "carbs_g": 8, "fat_g": 30},
                "calories": 470,
                "ingredients": [
                    {"name": "chicken breast", "qty_g": 150},
                    {"name": "mushrooms", "qty_g": 100},
                    {"name": "heavy cream", "qty_g": 50}
                ],
                "instructions": "Cook chicken breast, saute mushrooms, add heavy cream and simmer."
            },
            {
                "recipe_name": "Keto Avocado and Egg Salad",
                "macros": {"protein_g": 14, "carbs_g": 4, "fat_g": 32},
                "calories": 340,
                "ingredients": [
                    {"name": "avocado", "qty_g": 150},
                    {"name": "hard boiled egg", "qty_g": 100},
                    {"name": "mayonnaise", "qty_g": 15}
                ],
                "instructions": "Mash avocado, chop eggs, mix with mayonnaise and season."
            }
        ]
        
        matching_recipes = []
        for r in all_recipes:
            # Check exclusions
            excluded = False
            for exc in exclusions:
                for ing in r["ingredients"]:
                    if exc.lower() in ing["name"].lower():
                        excluded = True
                        break
                if exc.lower() in r["recipe_name"].lower():
                    excluded = True
                if excluded:
                    break
            if excluded:
                continue
                
            # Check calories ceiling
            if r["calories"] > target_calories:
                continue
                
            # Wrap using Recipe Pydantic model
            recipe_model = Recipe(
                recipe_name=r["recipe_name"],
                macros=UserMacros(
                    protein_g=r["macros"]["protein_g"],
                    carbs_g=r["macros"]["carbs_g"],
                    fat_g=r["macros"]["fat_g"]
                ),
                calories=float(r["calories"]),
                ingredients=[RecipeIngredient(name=x["name"], qty_g=float(x["qty_g"])) for x in r["ingredients"]],
                instructions=r["instructions"]
            )
            matching_recipes.append(recipe_model)
            
        if not matching_recipes:
            return FetchRecipesResponse(
                recipes=[],
                error="NO_RECIPES_FOUND",
                recovery_instruction="No recipes were found matching your constraints. Try raising the calories ceiling or reducing the exclusions or food dislikes."
            ).model_dump()
            
        return FetchRecipesResponse(
            recipes=matching_recipes
        ).model_dump()
        
    except Exception as e:
        logger.error(f"Error fetching recipes: {e}")
        return FetchRecipesResponse(
            recipes=[],
            error=str(e),
            recovery_instruction="Verify that target_calories is a positive integer, and that exclusions and focus_ingredients are lists of strings."
        ).model_dump()

# =====================================================================
# GUIDED ERROR HANDLING AND RECOVERY CALLBACKS
# =====================================================================

def guided_tool_error_handler(tool: Any, args: dict[str, Any], context: Any, error: Exception) -> dict[str, Any]:
    """Intercepts tool errors and returns structured JSON with guided recovery instructions."""
    error_msg = str(error)
    tool_name = tool.name if hasattr(tool, "name") else str(tool)
    logger.error(f"Guided Tool Error intercepted on '{tool_name}' with args {args}: {error_msg}")
    
    # Establish default guided instructions
    recovery_instruction = (
        "The system encountered an unexpected tool error. Please verify the arguments match "
        "the expected type specifications and that no parameters are missing, then retry."
    )
    
    # Customize recovery instructions based on tool name and specific error messages
    if "get_user_profile" in tool_name:
        recovery_instruction = (
            "The customer's profile could not be found or retrieved. Ask the user "
            "to provide their nutritional goals (daily calories, macro split percentages), "
            "and call update_user_profile to initialize their profile guidelines."
        )
    elif "update_user_profile" in tool_name:
        if "Macro" in error_msg or "percentage" in error_msg or "splits" in error_msg:
            recovery_instruction = (
                "Macronutrient splits are invalid because they do not sum to exactly 1.0 (or 100%). "
                "Ensure that fat_percent + protein_percent + carb_percent == 1.0 (or 100%). "
                "Request correct target splits from the user and call update_user_profile again."
            )
        elif "calorie" in error_msg or "range" in error_msg:
            recovery_instruction = (
                "The target daily calories are invalid (must be between 500 and 10000 kcal). "
                "Ask the user for a physically reasonable target calorie ceiling, then retry."
            )
    elif "query_pantry_supplies" in tool_name:
        recovery_instruction = (
            "No pantry inventory document exists for this customer. Inform them their "
            "pantry is currently empty, ask them to list their ingredients, and call "
            "update_pantry_supplies with operation='upsert' to populate their virtual pantry."
        )
    elif "update_pantry_supplies" in tool_name:
        recovery_instruction = (
            "Could not update pantry supplies. Please verify the inputs are valid: "
            "items must be a list of dictionaries with 'item' (string) and 'quantity_g' (float). "
            "Correct the formatting and re-attempt."
        )
    elif "fetch_recipes" in tool_name:
        recovery_instruction = (
            "No culinary recipes could be retrieved. Try relaxing search constraints "
            "by increasing the calorie ceiling, shortening the exclusion/dislikes list, "
            "or removing low-volume focus ingredients, then call fetch_recipes again."
        )
        
    return {
        "status": "error",
        "error_code": "TOOL_EXECUTION_FAILURE",
        "tool_name": tool_name,
        "error_message": error_msg,
        "recovery_instruction": recovery_instruction
    }

def guided_model_error_handler(context: Any, llm_request: Any, error: Exception) -> Any | None:
    """Intercepts model call errors and injects graceful recovery directions into the response."""
    error_msg = str(error)
    logger.error(f"Guided Model Error intercepted: {error_msg}")
    
    # Create an LlmResponse container with content instructing the agent on how to recover
    recovery_text = (
        f"The agent-model communication channel failed: {error_msg}. "
        "Recovery Directions: If this is a transient API rate-limiting or network error, please retry "
        "the operation. If it is a content safety block, please rephrase the request to be professional "
        "and strictly focused on macronutrient goals and recipe planning."
    )
    
    from google.adk.models.llm_response import LlmResponse
    from google.genai.types import Candidate, Content, Part
    
    try:
        return LlmResponse(
            content=Content(
                role="model",
                parts=[Part.from_text(text=recovery_text)]
            ),
            error_code="MODEL_EXECUTION_FAILURE",
            error_message=error_msg
        )
    except Exception as e:
        logger.error(f"Failed to build mock LlmResponse: {e}")
        return None

# =====================================================================
# 2. COLLABORATIVE SUBAGENTS (ADK 2.0 DECLARATIVE MODEL)
# =====================================================================

async def log_before_agent(callback_context: CallbackContext) -> None:
    """Logs the entry state of an agent with standard trace contexts."""
    logger.info(
        f"Entering Agent: {callback_context.agent_name}",
        extra={
            "user_id": callback_context.user_id,
            "trace_id": callback_context.run_id,
            "session_id": callback_context.session.id if callback_context.session else None,
            "agent_name": callback_context.agent_name,
            "event": "agent_enter"
        }
    )

async def log_after_agent(callback_context: CallbackContext) -> None:
    """Logs the exit state of an agent with standard trace contexts."""
    logger.info(
        f"Exiting Agent: {callback_context.agent_name}",
        extra={
            "user_id": callback_context.user_id,
            "trace_id": callback_context.run_id,
            "session_id": callback_context.session.id if callback_context.session else None,
            "agent_name": callback_context.agent_name,
            "event": "agent_exit"
        }
    )

diet_preferences_agent = Agent(
    name="diet_preferences_agent",
    mode="task",
    input_schema=DietPreferencesInput,
    output_schema=DietPreferencesOutput,
    description="Manages user nutritional targets, macronutrient calculations, likes, dislikes, and allergens.",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=(
        "You are an expert nutritionist assistant. Your sole responsibility is managing user nutritional targets, "
        "calculating healthy macronutrient allocations, and tracking active food likes, dislikes, and allergens. "
        "Use the provided tools to query or update profiles. Always perform any calorie-to-gram math accurately. "
        "Upon completing your work, return the final result by calling 'finish_task' with the specified output schema."
    ),
    tools=[get_user_profile, update_user_profile],
    before_agent_callback=log_before_agent,
    after_agent_callback=log_after_agent,
    on_tool_error_callback=guided_tool_error_handler,
    on_model_error_callback=guided_model_error_handler
)

pantry_supply_agent = Agent(
    name="pantry_supply_agent",
    mode="task",
    input_schema=PantrySupplyInput,
    output_schema=PantrySupplyOutput,
    description="Audits virtual pantry inventory levels, ingredient volumes, and tracks impending expiration timelines.",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=(
        "You are an expert pantry supervisor. Your responsibility is to audit inventory levels, estimate ingredient "
        "volumes from natural descriptions, and note impending expiration dates. Use the pantry tools to read/write state. "
        "Upon completing your work, return the final result by calling 'finish_task' with the specified output schema."
    ),
    tools=[query_pantry_supplies, update_pantry_supplies],
    before_agent_callback=log_before_agent,
    after_agent_callback=log_after_agent,
    on_tool_error_callback=guided_tool_error_handler,
    on_model_error_callback=guided_model_error_handler
)

meal_planner_agent = Agent(
    name="meal_planner_agent",
    mode="task",
    input_schema=MealPlannerInput,
    output_schema=MealPlannerOutput,
    description="Queries, filters, and matches culinary recipes to plan balanced meals using available stocks; compiles missing items.",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=(
        "You are a master culinary planner. Analyze target caloric limits, ingredient dislikes, and available "
        "pantry items. Match them to high-quality recipes. Compute missing components to generate a logical shopping list. "
        "Upon completing your work, return the final result by calling 'finish_task' with the specified output schema."
    ),
    tools=[fetch_recipes],
    before_agent_callback=log_before_agent,
    after_agent_callback=log_after_agent,
    on_tool_error_callback=guided_tool_error_handler,
    on_model_error_callback=guided_model_error_handler
)

# =====================================================================
# 3. ROOT COORDINATOR (GRAPH ENTRY POINT & ROUTING)
# =====================================================================

async def root_after_agent_callback(callback_context: CallbackContext) -> None:
    """Combines transition exit tracing and cross-session memory generation."""
    await log_after_agent(callback_context)
    logger.info("Triggering cross-session memory generation.")
    try:
        await callback_context.add_session_to_memory()
    except Exception as e:
        logger.error(f"Failed to generate memories: {e}")

root_agent = Agent(
    name="nutrition_orchestrator",
    model=Gemini(
        model="gemini-2.5-pro",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=(
        "You are the central coordinator for the Nutrition & Pantry Assistant. "
        "Your role is to orchestrate tasks sequentially among your specialized subagents:\n"
        "1. Delegate preference and profile management to diet_preferences_agent by requesting a task.\n"
        "2. Delegate inventory auditing and pantry checks to pantry_supply_agent by requesting a task.\n"
        "3. Delegate cooking recipe matches and missing ingredients compilation to meal_planner_agent by requesting a task.\n"
        "4. Combine these typed results and output a highly organized, beautifully formatted Weekly Meal Plan and structured Grocery List.\n"
        "Never perform tasks yourself that should be handled by a specialized subagent."
    ),
    sub_agents=[diet_preferences_agent, pantry_supply_agent, meal_planner_agent],
    tools=[PreloadMemoryTool()],
    before_agent_callback=log_before_agent,
    after_agent_callback=root_after_agent_callback,
    on_tool_error_callback=guided_tool_error_handler,
    on_model_error_callback=guided_model_error_handler
)

# Packaging the agent execution graph into a deployable App instance
app = App(
    root_agent=root_agent,
    name="app",
)
