def extract_thinking_process(response):
    """Extract thinking process from response"""
    # Some responses (Open-Reasoner-Zero models) contain an instructional preamble like "<think> </think>" before the real reasoning.
    # We therefore take the *last* "<think>" and its corresponding closing tag.
    think_start = response.rfind("<think>")
    if think_start == -1:
        think_start = 0
    else:
        think_start += len("<think>")

    think_end = response.find("</think>", think_start)
    if think_end == -1:
        think_end = len(response)

    return response[think_start:think_end].strip()