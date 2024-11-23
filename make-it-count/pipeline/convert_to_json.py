import json
import spacy
import re

word2number = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
    'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20,
    'twenty-one': 21, 'twenty-two': 22, 'twenty-three': 23, 'twenty-four': 24, 'twenty-five': 25
}

def convert_number_word_to_int(text):
    """Convert both word numbers and digits to integers"""
    text = text.lower()
    if text.isdigit():
        return int(text)
    return word2number.get(text, 0)

# Updated pattern to handle both words and digits
pattern = re.compile(r'\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|twenty-one|twenty-two|twenty-three|twenty-four|twenty-five|\d+)\s(\w+)')

prompts_list = [
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
prompts = [
    {
        "prompt": prompt,
        "object": match.group(2),
        "int_number": convert_number_word_to_int(match.group(1)),
        "seed": 1
    }
    for prompt in prompts_list
    if (match := pattern.search(prompt.lower()))
]

print(prompts)
# Filter only prompts with numbers ≤ 9 since that's what the model can handle
prompts = [p for p in prompts if p["int_number"] <= 25]

with open("data_count.json", "w") as f:
    json.dump(prompts, f, indent=2)