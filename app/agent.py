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
from typing import List, Dict, Any

from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import Gemini
from google.genai import types
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.preload_memory_tool import PreloadMemoryTool
from google.adk.tools import ToolContext

# Setup PII Scrubbing and Structured Logging
EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_REGEX = re.compile(r"\b(?:\+?(\d{1,3}))?[-. (]*(\d{3})[-. )]*(\d{3})[-. ]*(\d{4})\b")

def scrub_pii(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    # Scrub email addresses
    text = EMAIL_REGEX.sub("[REDACTED_EMAIL]", text)
    # Scrub telephone numbers
    text = PHONE_REGEX.sub("[REDACTED_PHONE]", text)
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
        A dictionary containing calorie goals, gram target splits, and dietary restrictions.
    """
    logger.info(f"Retrieving user profile for user: {user_id}")
    doc_ref = db.collection("user_profiles").document(user_id)
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict()
    return {}

def update_user_profile(user_id: str, profile_data: Dict[str, Any], tool_context: ToolContext = None) -> bool:
    """Updates the user's dietary preferences and target metrics in Firestore.
    
    If calorie and macro percentage split targets are provided, the corresponding
    grams are calculated as:
    - Fat: (calories * fat_percent) / 9
    - Protein: (calories * protein_percent) / 4
    - Carbohydrates: (calories * carb_percent) / 4
    
    Args:
        user_id: The unique system identifier for the customer.
        profile_data: Key-value updates (e.g., target calories, new disliked foods, macro split percentages).
        tool_context: Optional context for human-in-the-loop validation and approvals.
        
    Returns:
        True if the write transaction completed successfully, False otherwise.
    """
    if tool_context is not None:
        if not tool_context.tool_confirmation or not tool_context.tool_confirmation.confirmed:
            tool_context.request_confirmation(
                hint=f"Please confirm updating your dietary profile guidelines with: {profile_data}"
            )
            return False

    logger.info(f"Updating user profile for user: {user_id} with data: {profile_data}")
    
    # Calculate grams if percent splits and calories are specified
    if "calories" in profile_data and "macro_splits" in profile_data:
        try:
            cals = float(profile_data["calories"])
            splits = profile_data["macro_splits"] # e.g., {"fat_percent": 0.70, "protein_percent": 0.25, "carb_percent": 0.05}
            fat_pct = float(splits.get("fat_percent", 0))
            prot_pct = float(splits.get("protein_percent", 0))
            carb_pct = float(splits.get("carb_percent", 0))
            
            # Math:
            # Fat: 9 calories/gram
            # Protein: 4 calories/gram
            # Carbs: 4 calories/gram
            fat_g = round((cals * fat_pct) / 9, 1)
            protein_g = round((cals * prot_pct) / 4, 1)
            carbs_g = round((cals * carb_pct) / 4, 1)
            
            profile_data["macros"] = {
                "fat_g": fat_g,
                "protein_g": protein_g,
                "carbs_g": carbs_g
            }
        except Exception as e:
            logger.error(f"Error calculating macro weights: {e}")
            
    # Load current and update
    current = get_user_profile(user_id)
    current.update(profile_data)
    
    # Store strictly to Firestore
    db.collection("user_profiles").document(user_id).set(current)
    return True

def query_pantry_supplies(user_id: str) -> List[Dict[str, Any]]:
    """Retrieves a list of current ingredients available in the user's pantry.
    
    Args:
        user_id: The unique system identifier for the customer.
        
    Returns:
        A list of active food items, each showing quantity (grams/units) and estimated expiration timeline.
    """
    logger.info(f"Retrieving pantry supplies for user: {user_id}")
    doc_ref = db.collection("pantry_supplies").document(user_id)
    doc = doc_ref.get()
    if doc.exists:
        data = doc.to_dict()
        return data.get("items", [])
    return []

def update_pantry_supplies(user_id: str, items: List[Dict[str, Any]], operation: str = "upsert", tool_context: ToolContext = None) -> bool:
    """Updates pantry inventory by adding purchased items or subtracting consumed items.
    
    Args:
        user_id: The unique system identifier for the customer.
        items: List of dictionary records containing "item" and "quantity_g".
        operation: 'upsert' to add/increment stock, 'consume' or 'delete' to subtract or remove.
        tool_context: Optional context for human-in-the-loop validation and approvals.
        
    Returns:
        True if the database transaction completed successfully, False otherwise.
    """
    if tool_context is not None:
        if not tool_context.tool_confirmation or not tool_context.tool_confirmation.confirmed:
            tool_context.request_confirmation(
                hint=f"Please confirm updating your pantry supplies: {operation}ing items: {items}"
            )
            return False

    logger.info(f"Updating pantry supplies for user: {user_id} with items: {items}, operation: {operation}")
    
    current_items = query_pantry_supplies(user_id)
    pantry_dict = {x["item"].lower(): x for x in current_items}
    
    for update_item in items:
        name = update_item["item"].lower()
        qty = float(update_item.get("quantity_g", 0))
        
        if operation == "upsert":
            if name in pantry_dict:
                pantry_dict[name]["quantity_g"] = pantry_dict[name].get("quantity_g", 0) + qty
            else:
                pantry_dict[name] = {
                    "item": update_item["item"],
                    "quantity_g": qty,
                    "expiry_days": update_item.get("expiry_days", 7)
                }
        elif operation in ("consume", "delete"):
            if name in pantry_dict:
                current_qty = pantry_dict[name].get("quantity_g", 0)
                if current_qty <= qty or qty == 0:
                    del pantry_dict[name]
                else:
                    pantry_dict[name]["quantity_g"] = current_qty - qty
                    
    updated_list = list(pantry_dict.values())
    
    # Store strictly to Firestore
    db.collection("pantry_supplies").document(user_id).set({"items": updated_list})
    return True

def fetch_recipes(target_calories: int, exclusions: List[str], focus_ingredients: List[str]) -> List[Dict[str, Any]]:
    """Queries a vectorized catalog for recipes using in-stock items and avoiding exclusions.
    
    Args:
        target_calories: Target per-meal caloric ceiling.
        exclusions: List of ingredients to strictly avoid.
        focus_ingredients: Stock ingredients to prioritize using first.
        
    Returns:
        A list of compatible recipes complete with directions, ingredient weights, and macro values.
    """
    logger.info(f"Fetching recipes matching target_calories: {target_calories}, exclusions: {exclusions}, focus_ingredients: {focus_ingredients}")
    
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
            
        matching_recipes.append(r)
        
    return matching_recipes

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
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=(
        "You are an expert nutritionist assistant. Your sole responsibility is managing user nutritional targets, "
        "calculating healthy macronutrient allocations, and tracking active food likes, dislikes, and allergens. "
        "Use the provided tools to query or update profiles. Always perform any calorie-to-gram math accurately."
    ),
    tools=[get_user_profile, update_user_profile],
    before_agent_callback=log_before_agent,
    after_agent_callback=log_after_agent
)

pantry_supply_agent = Agent(
    name="pantry_supply_agent",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=(
        "You are an expert pantry supervisor. Your responsibility is to audit inventory levels, estimate ingredient "
        "volumes from natural descriptions, and note impending expiration dates. Use the pantry tools to read/write state."
    ),
    tools=[query_pantry_supplies, update_pantry_supplies],
    before_agent_callback=log_before_agent,
    after_agent_callback=log_after_agent
)

meal_planner_agent = Agent(
    name="meal_planner_agent",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=(
        "You are a master culinary planner. Analyze target caloric limits, ingredient dislikes, and available "
        "pantry items. Match them to high-quality recipes. Compute missing components to generate a logical shopping list."
    ),
    tools=[fetch_recipes],
    before_agent_callback=log_before_agent,
    after_agent_callback=log_after_agent
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
        model="gemini-3.5-flash",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=(
        "You are the central coordinator for the Nutrition & Pantry Assistant. "
        "Your role is to orchestrate tasks sequentially among your specialized subagents:\n"
        "1. Extract current preferences, calorie goals, and food restrictions using diet_preferences_agent.\n"
        "2. Audit available ingredients currently in stock using pantry_supply_agent.\n"
        "3. Deliver these data boundaries to meal_planner_agent to search matching recipes and isolate missing components.\n"
        "4. Output a highly organized, beautifully formatted Weekly Meal Plan and organized Grocery List.\n"
        "Never perform tasks yourself that should be handled by a specialized subagent."
    ),
    sub_agents=[diet_preferences_agent, pantry_supply_agent, meal_planner_agent],
    tools=[PreloadMemoryTool()],
    before_agent_callback=log_before_agent,
    after_agent_callback=root_after_agent_callback
)

# Packaging the agent execution graph into a deployable App instance
app = App(
    root_agent=root_agent,
    name="app",
)
