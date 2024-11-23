# The trailing space and "\n" is not a problem since they will be removed before making an API request.
templatev0_1 = """You are an intelligent bounding box generator. I will provide you with a caption for a photo, image, or painting. Your task is to generate the bounding boxes for the objects mentioned in the caption, along with a background prompt describing the scene. The images are of size 512x512. The top-left corner has coordinate [0, 0]. The bottom-right corner has coordinnate [512, 512]. The bounding boxes should not overlap or go beyond the image boundaries. Each bounding box should be in the format of (object name, [top-left x coordinate, top-left y coordinate, box width, box height]) and should not include more than one object. Do not put objects that are already provided in the bounding boxes into the background prompt. Do not include non-existing or excluded objects in the background prompt. Use "A realistic scene" as the background prompt if no background is given in the prompt. If needed, you can make reasonable guesses. Please refer to the example below for the desired format. Try to use valid characters. For the **object name** keep the fractional part as well if it is mentioned in the prompt for example: If it is mentioned half a pizza then object name will be **half of a pizza** not pizza.

Caption: A realistic image of landscape scene depicting a green car parking on the left of a blue truck, with a red air balloon and a bird in the sky
Objects: [('a green car', [21, 281, 211, 159]), ('a blue truck', [269, 283, 209, 160]), ('a red air balloon', [66, 8, 145, 135]), ('a bird', [296, 42, 143, 100])]
Background prompt: A realistic landscape scene
Negative prompt: 

Caption: Depict a sandwich with third of it removed
Objects: [('two third of a sandwich', [106, 200, 300, 212])]
Background prompt: A realistic scene
Negative prompt:

Caption: Show a jar of candies filled up to four-fifths of its capacity
Objects: [('a jar of candies filled up to four fifths', [106, 150, 300, 212])]
Background prompt: A realistic scene
Negative prompt:

Caption: Show a pie chart divided into three equal parts
Objects: [('a pie chart divided into three equal parts', [106, 150, 300, 212])]
Background prompt: A realistic scene
Negative prompt:

Caption: Create an image of a pie with three-fourths of it remaining
Objects: [('three-fourth of a pie', [106, 150, 300, 212])]
Background prompt: A realistic scene
Negative prompt:

Caption: Depict a set of eight apples, with one of them partially eaten
Objects: [('an apple', [50, 200, 100, 100]), ('an apple', [160, 200, 100, 100]), ('an apple', [270, 200, 100, 100]), ('an apple', [380, 200, 100, 100]), ('an apple', [50, 320, 100, 100]), ('an apple', [160, 320, 100, 100]), ('an apple', [270, 320, 100, 100]), ('a partially eaten apple', [380, 320, 100, 100])]
Background prompt: A realistic scene
Negative prompt:

Caption: A realistic top-down view of a wooden table with two apples on it
Objects: [('a wooden table', [20, 148, 472, 216]), ('an apple', [150, 226, 100, 100]), ('an apple', [280, 226, 100, 100])]
Background prompt: A realistic top-down view
Negative prompt: 

Caption: A realistic scene of three skiers standing in a line on the snow near a palm tree
Objects: [('a skier', [5, 152, 139, 168]), ('a skier', [278, 192, 121, 158]), ('a skier', [148, 173, 124, 155]), ('a palm tree', [404, 105, 103, 251])]
Background prompt: A realistic outdoor scene with snow
Negative prompt: 

Caption: An oil painting of a pink dolphin jumping on the left of a steam boat on the sea
Objects: [('a steam boat', [232, 225, 257, 149]), ('a jumping pink dolphin', [21, 249, 189, 123])]
Background prompt: An oil painting of the sea
Negative prompt: 

Caption: A cute cat and an angry dog without birds
Objects: [('a cute cat', [51, 67, 271, 324]), ('an angry dog', [302, 119, 211, 228])]
Background prompt: A realistic scene
Negative prompt: birds

Caption: Two pandas in a forest without flowers
Objects: [('a panda', [30, 171, 212, 226]), ('a panda', [264, 173, 222, 221])]
Background prompt: A forest
Negative prompt: flowers

Caption: An oil painting of a living room scene without chairs with a painting mounted on the wall, a cabinet below the painting, and two flower vases on the cabinet
Objects: [('a painting', [88, 85, 335, 203]), ('a cabinet', [57, 308, 404, 201]), ('a flower vase', [166, 222, 92, 108]), ('a flower vase', [328, 222, 92, 108])]
Background prompt: An oil painting of a living room scene
Negative prompt: chairs

Caption: {prompt}
Objects: 
"""

DEFAULT_SO_NEGATIVE_PROMPT = "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image, bad proportions, duplicate, two, many, group, occlusion, occluded, side, border, collate"
DEFAULT_OVERALL_NEGATIVE_PROMPT = "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image, bad proportions, duplicate"

templates = {"v0.1": templatev0_1}
template_versions = ["v0.1"]

stop = "\n\n"


prompts_demo_gpt4, prompts_demo_gpt3_5 = [], []

# Put what we want to generate when you query GPT-4 for demo here
prompts_demo_gpt4 = [
        "Create an image of exactly six red and two blue balloons in a garden.",
    "Generate an image with exactly eleven pencils arranged neatly on a desk.",
    "Create an image with exactly six ducks swimming in a lake.",
    "Depict an image of exactly two cars parked on a street.",
    "Create an image with exactly six fruits in a bowl: two apples, three bananas, and one orange.",
    "Generate an image of five children playing on a seesaw.",
    "Create an image of exactly six trees in a park with a clear path in the middle.",
    "Show an image with exactly seven  kittens on a sofa.",
    "Create an image of exactly seven candles lit on a dining table, with five wine glasses beside them.",
    "Depict a scene with exactly four puppies playing on a grassy field near a small pond.",
    "Show an image of exactly fourteen birds sitting on two telephone wires.",
    "Create an image of exactly fifteen bicycles parked in a row outside a library.",
    "Depict a beach scene with exactly six people building sandcastles near the shoreline.",
    "Generate an image of exactly two horses standing in a stable, with a pile of five haystacks beside them.",
    "Generate an image of a classroom with exactly fifteen students sitting at their desks.",
    "Create an image of exactly two dogs playing in a park, with four children watching them.",
    "Depict a scene with exactly five cars parked in a parking lot with two tree, and one bicycles nearby.",
    "Show an image of exactly nine birds perched on a fence, with three more flying in the sky above.",
    "Generate an image of a garden with exactly eight sunflowers in full bloom, surrounded by three rose bushes.",
    "Create an image of exactly three dogs sleeping on a sofa, with six cushions arranged neatly on the floor.",
    "Show an image of exactly two boats sailing on a river, with one ducks swimming alongside them.",
    "Depict a scene with exactly three kites flying in the sky, with four children running to catch them.",
    "Create an image of exactly five balloons tied to a fence, with five more floating in the sky above.",
    "Show an image of exactly two ice cream trucks parked on a street, with seven children lined up to buy ice cream.",
    "Create an image of exactly five pumpkins arranged on a porch, with three skeletons standing in the yard.",
    "Show an image of exactly one airplanes on an airport runway, with seventeen helicopters hovering nearby.",
    "Create an image of exactly five geese swimming in a pond",
    "Generate an image of a kitchen counter with exactly fifteen loaves of bread, stacked neatly in five piles.",
    "Depict a scene with exactly four children blowing bubbles.",
    "Show an image of exactly two firefighters using hoses to spray water.",
    "Create an image of exactly five goats standing on a rocky hill, with three more grazing on the grass below.",
    "Generate an image of a library with exactly six people reading books, and seven books stacked on a nearby table.",
    "Depict a scene with exactly five hot air balloons floating in the sky, with six clouds drifting nearby.",
    "Create an image of exactly three astronauts standing beside one space shuttles on a launchpad.",
    "Generate an image of a farmyard with exactly six cows grazing and one chickens pecking the ground nearby.",
    "Show an image of exactly four school buses lined up in front of a school, with twelve children boarding one of them.",
    "Create an image of a living room with exactly five pillows on a sofa and three magazines spread on a coffee table.",
    "Depict an image of exactly three lions resting in a savannah, with four gazelles grazing in the background.",
    "Show an image of a beach with exactly four surfboards stuck in the sand and five seagulls walking nearby.",
    "Create an image of a boardroom with exactly eleven chairs around a table and three monitors open on the table.",
    "Generate an image of a circus tent with exactly six clowns swinging.",
    "Depict a scene with exactly four firefighters climbing a ladder and nine more waiting at the bottom.",
    "Generate an image of exactly six rabbits hopping in a meadow beside a large tree.",
    "Depict an image of a concert with exactly six musicians playing instruments.",
    "Create an image of exactly six ice cream cones displayed on a cart, with one child holding a cone in his hand.",
    "Show an image of a farm with exactly eight sheep grazing in the field, with two resting near a fence.",
    "Generate an image of a picnic scene with exactly five sandwiches laid on a blanket and four people holding sandwiches.",
    "Show an image of a carnival booth with exactly four teddy bears on display.",
    "Create an image of exactly five fire trucks parked nearby.",
    "Create an image of exactly seven mailboxes lined up on a street.",
    "Generate an image of exactly four musicians playing instruments in a park.",
    "Depict a scene with exactly five horses galloping in a field.",
    "Generate an image of a bus stop with exactly six people waiting.",
    "Show a farmyard with exactly five pigs playing in the mud, with three chickens standing nearby.",
    "Create an image of exactly six students sitting in a line, with six books open in front of them.",
    "Generate an image of a riverbank with exactly four herons standing in the shallow water, with four turtles sunbathing on a nearby rock.",
    "Show an image of a classroom where exactly three children are holding paintbrushes, with ten paintings.",
    "Create an image of exactly eight oranges arranged in a basket, with five bananas lying inside the basket on a wooden table.",
    "Depict a train station platform with exactly three suitcases arranged in a line, while three passengers are waiting beside them.",
    "Generate an image of a campsite with exactly three tents set up in a circle, with one campfire burning in front of them.",
    "Show a football field with exactly six players standing on the ground.",
    "Create an image of a restaurant counter with exactly four plates arranged on it, while five chefs are behind the counter.",
    "Depict a garden with exactly six rabbits hopping on the grass.",
    "Create an image of exactly ten baskets of strawberries at a farmer\u2019s market, with three persons inspecting the fruit.",
    "Generate an image of exactly six ducks waddling in a farmyard.",
    "Depict a scene with exactly five kayaks floating on a river.",
    "Create an image of exactly four parrots sitting on branches in a rainforest.",
    "Generate an image of exactly seven hikers trekking on a trail.",
    "Show an image of a music room with exactly ten guitars hanging on a wall, and six amplifiers on the floor below.",
    "Create an image of exactly sixteen seashells arranged on a sandy beach."
]

# Put what we want to generate when you query GPT-3.5 for demo here
prompts_demo_gpt3_5 = []

prompt_types = [
    "demo",
    "lmd_negation",
    "lmd_numeracy",
    "lmd_attribution",
    "lmd_spatial",
    "lmd",
]


def get_prompts(prompt_type, model, allow_non_exist=False):
    """
    This function returns the text prompts according to the requested `prompt_type` and `model`. Set `model` to "all" to return all the text prompts in the current type. Otherwise we may want to have different prompts for gpt-3.5 and gpt-4 to prevent confusion.
    """
    prompts_gpt4, prompts_gpt3_5 = {}, {}
    if prompt_type.startswith("lmd"):
        from utils.eval.lmd import get_lmd_prompts

        prompts = get_lmd_prompts()

        # We do not add to both dict to prevent duplicates when model is set to "all".
        if "gpt-4" in model:
            prompts_gpt4.update(prompts)
        else:
            prompts_gpt3_5.update(prompts)
    elif prompt_type == "demo":
        prompts_gpt4["demo"] = prompts_demo_gpt4
        prompts_gpt3_5["demo"] = prompts_demo_gpt3_5

    if "all" in model:
        return prompts_gpt4.get(prompt_type, []) + prompts_gpt3_5.get(prompt_type, [])
    elif "gpt-4" in model:
        if allow_non_exist:
            return prompts_gpt4.get(prompt_type, [])
        return prompts_gpt4[prompt_type]
    else:
        # Default: gpt-3.5
        if allow_non_exist:
            return prompts_gpt3_5.get(prompt_type, [])
        return prompts_gpt3_5[prompt_type]


if __name__ == "__main__":
    # Print the full prompt for the latest prompt in prompts
    # This allows pasting into an LLM web UI
    prompt_type = "demo"

    assert prompt_type in prompt_types, f"prompt_type {prompt_type} does not exist"

    prompts = get_prompts(prompt_type, "all")
    prompt = prompts[-1]

    prompt_full = templatev0_1.format(prompt=prompt.strip().rstrip("."))
    print(prompt_full)

    if False:
        # Useful if you want to query an LLM with JSON input
        
        import json

        print(json.dumps(prompt_full.strip("\n")))
