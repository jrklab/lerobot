class CommandParser:
    def __init__(self):
        self.base_commands = {
            "forward": ["forward"],
            "backward": ["reverse", "backward"],
            "rotate_left": ["rotate left", "turn left", "rotate counter clockwise"],
            "rotate_right": ["rotate right", "turn right", "rotate clockwise"],
            "left": ["left"],
            "right": ["right"],
            "speed_up": ["speed up"],
            "speed_down": ["speed down"],
            "stop": ["stop", "quit", "terminate"],
            "continuous_mode": ["continuous mode", "continuous"],
            "decay_mode": ["decay mode", "decay"],
        }

        self.head_commands = {
            "up": ["up"],
            "down": ["down"],
            "turn_left": ["turn left"],
            "turn_right": ["turn right"],
            "reset": ["reset"],
        }
        
        self.all_commands = {
            "base": self.base_commands,
            "head": self.head_commands,
        }

    def parse_command(self, text):
        """
        Parses the text to find a matching command.
        Args:
            text (str): The text to parse.
        Returns:
            tuple or None: A tuple representing the command, e.g., ('base', 'forward'), or None if no command is found.
        """
        text = text.lower()

        # Prioritize commands with explicit context
        if 'base' in text and 'head' not in text:
            components_to_check = ['base', 'head']
        elif 'head' in text and 'base' not in text:
            components_to_check = ['head', 'base']
        else: # no context or both contexts, default to head for ambiguous turns
            components_to_check = ['head', 'base'] 

        for component in components_to_check:
            commands = self.all_commands[component]
            # Make sure longer keywords are checked first to handle substrings
            sorted_commands = sorted(commands.items(), key=lambda x: max(len(k) for k in x[1]), reverse=True)

            for command, keywords in sorted_commands:
                for keyword in keywords:
                    is_match = False
                    if ' ' in keyword:
                        if all(word in text for word in keyword.split()):
                            is_match = True
                    elif keyword in text:
                        is_match = True
                    
                    if is_match:
                        return (component, command)
        return None

if __name__ == '__main__':
    parser = CommandParser()
    
    test_phrases = [
        "base forward",
        "go backward",
        "turn the base to the left",
        "head up",
        "head down",
        "could you please turn the head to the right",
        "reset head position",
        "speed up a little",
        "this is not a command",
        "rotate left",
        "base rotate left",
        "left",
        "base left",
        "rotate right",
        "base rotate right",
        "right",
        "base right",
        "rotate clockwise",
        "base rotate counter clockwise",
        "stop the base",
        "quit now",
        "continuous mode",
        "decay",
    ]

    for phrase in test_phrases:
        command = parser.parse_command(phrase)
        print(f"'{phrase}' -> {command}")
