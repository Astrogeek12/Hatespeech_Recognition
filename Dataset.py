import pandas as pd
import random

# Define hate speech and normal comments
hate_comments = ["You're an idiot!", "Go back to your country!", "I hate you!", "You're worthless!",
                 "This is disgusting!", "You should die!", "You're a disgrace!", "You're pathetic!",
                 "You're a waste of space!", "I hope you suffer!","You're inferior because of your race/ethnicity/religion.",
                 "Go back to where you came from!","You deserve to be discriminated against because of your gender/sexual orientation/disability.","I wish [group of people] didn't exist.",
                 "You're a [derogatory slur] and don't belong here.","You're disgusting because of your appearance/identity.","I hope you get hurt because of who you are.",
                 "You're less of a person because of your beliefs/ideologies.","You're not welcome here because of your background.","You deserve to be [violence or harm] because of your identity."]
normal_comments = ["What a lovely day!", "I like your outfit!", "Great job!", "Nice work!", "Thank you!",
                   "Have a nice day!", "You're awesome!", "I appreciate your help.", "Congratulations!",
                   "You're the best!","That's great","Pretty awesome","Keep goimg","Love your performance","Shining star"]

# Generate the table
data = []
for _ in range(200):
    class_label = random.choice(["hate_speech", "normal"])
    if class_label == "hate_speech":
        comment = random.choice(hate_comments)
    else:
        comment = random.choice(normal_comments)
    data.append((class_label, comment))

# Create DataFrame
df = pd.DataFrame(data, columns=["class", "comment"])

# Save DataFrame to CSV file
df.to_csv("comments_dataset.csv", index=False)

print("CSV file saved successfully.")
