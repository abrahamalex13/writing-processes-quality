import numpy as np
import pandas as pd

def extract(path):

    X = pd.read_csv(path)
    X = X.sort_values(["id", "event_id"], ascending=[True, True])
    
    return X

def scrub_activity(X):

    # 'Move From' activity recorded with low-level cursor loc details
    # extract bigger-picture 'Move From'
    # QUESTION: what's the difference between Move From, and a cut+paste?
    X['activity_detailed'] = X['activity']
    X.loc[X['activity'].str.contains('Move From'), 'activity'] = 'Move'

    return X

def scrub_text_change(X):
    """
    Problems with initial text data:

    - Some hex expressions (\\xHH) not decoded. Instead, written literally.
        - Examples: emdash (\\x96), slanted quotations & ticks.
        
    - Some foreign characters (accent a, overring a) not anonymized with generic q.
    Problem confirmed via Kaggle data viewer, for id-event_id cases like 
    0916cdad-39 or 9f328eb3-19. Solutions:
        - An Input event cannot include multiple characters: 
        foreign character & something else. 
        Then, 
            - If Input event contains any emdash, overwrite as strictly emdash
            - If Input event contains no emdash & foreign character, overwrite with single q
            - If Move event, replace any foreign character with single q
    """

    X['text_change_original'] = X['text_change']

    # expect this transforms all \xHH literals
    X['text_change'] = (
        X
        ['text_change_original']
        # arrived at utf-8 encode, windows-1252 decode after several iterations.
        # tested latin-1, but not all \xHH instances caught.
        # tested utf-16, just rose errors.
        .apply(lambda x: x.encode(encoding='utf-8').decode("windows-1252"))
    )


    is_text_change_decode_english = (
        X['text_change'].apply(lambda x: x.isascii())
    )

    is_input_event_foreign_any_emdash = (
        (~ is_text_change_decode_english)
        & (X['activity'] == "Input") 
        & (X['text_change'].str.contains("—"))
    )
    X.loc[is_input_event_foreign_any_emdash, 'text_change'] = "—"

    is_input_event_foreign_no_overwrite = (
        (~ is_text_change_decode_english)
        & (X['activity'] == "Input")
        & (~ X['text_change'].str.contains("—"))
    )
    X.loc[is_input_event_foreign_no_overwrite, 'text_change'] = 'q'


    # given block text change, proceed one character at a time,
    # replacing foreign ones 
    def anonymize_non_ascii(x):
        value = ""
        for x_i in x:
            if not x_i.isascii():
                value += "q"
            else:
                value += x_i
        return value

    X['text_change'] = np.where(
        X['activity'].str.contains('Move|Remove|Paste|Replace', regex=True),
        X['text_change'].apply(lambda x: anonymize_non_ascii(x)),
        X['text_change']
    )

    return X

def concatenate_essay_from_logs(df):
    """
    Concatenate essay text from disparate logged input events.
    Expect df to be *one* author's log.
    Adapted from sources: 
        https://www.kaggle.com/code/hiarsl/feature-engineering-sentence-paragraph-features,
        https://www.kaggle.com/code/kawaiicoderuwu/essay-contructor.
    """

    input_events = df.loc[
        (df.activity != 'Nonproduction'), 
        ['activity_detailed', 'cursor_position', 'text_change']
        ].rename(columns={'activity_detailed': 'activity'})

    essay_text = ""
    for input_event in input_events.values:

        activity = input_event[0]
        cursor_position_after_event = input_event[1]
        text_change_log = input_event[2]

        if activity == 'Replace':

            replace_from_to = text_change_log.split(' => ')
            text_add = replace_from_to[1]
            text_remove = replace_from_to[0]
            cursor_position_start_text_change = (
                cursor_position_after_event - len(text_add)
                )
            cursor_position_after_skip_replace = (
                cursor_position_start_text_change + len(text_remove)
            )

            # essayText start: "the blue cat"
            # replace "blue" with "red"
            # "the redblue cat", skip blue
            essay_text = (
                essay_text[:cursor_position_start_text_change] # "the "
                + text_add # "red"
                # essayText value: "the blue cat" 
                # want remaining " cat", NOT "blue cat"
                + essay_text[cursor_position_after_skip_replace:] 
                )

            continue

        if activity == 'Paste':

            cursor_position_start_text_change = (
                cursor_position_after_event - len(text_change_log)
                )

            # essayText start: "the cat"
            # paste "blue " between
            essay_text = (
                essay_text[:cursor_position_start_text_change] # "the " 
                + text_change_log # "blue "
                # essayText value: "the cat"
                + essay_text[cursor_position_start_text_change:]
            )

            continue

        if activity == 'Remove/Cut':
            # similar process to "Replace" action

            text_remove = text_change_log
            cursor_position_after_skip_remove = (
                cursor_position_after_event + len(text_remove)
            )

            essay_text = (
                essay_text[:cursor_position_after_event] 
                + essay_text[cursor_position_after_skip_remove:]
                )

            continue
        
        if "Move" in activity:

            cursor_intervals_raw_str = (
                activity[10:]
                .replace("[", "")
                .replace("]", "")
                )
            cursor_intervals_separate = cursor_intervals_raw_str.split(' To ')
            cursor_intervals_vectors = [
                x.split(', ') 
                for x in cursor_intervals_separate
                ]
            cursor_interval_from = [
                int(x) for x in cursor_intervals_vectors[0]
                ]
            cursor_interval_to = [
                int(x) for x in cursor_intervals_vectors[1]
                ]

            # "the blue cat ran", move "blue" to
            # "the cat blue ran"
            # note: no change in total text length

            if cursor_interval_from[0] != cursor_interval_to[0]:

                if cursor_interval_from[0] < cursor_interval_to[0]:
                    
                    essay_text = (
                        # all text preceding move-impacted window
                        essay_text[:cursor_interval_from[0]] +
                        # skip where moved block _was_,
                        # proceed to end of move-impacted window
                        essay_text[cursor_interval_from[1]:cursor_interval_to[1]] +
                        # add moved block
                        essay_text[cursor_interval_from[0]:cursor_interval_from[1]] + 
                        # all text proceeding move-impacted window
                        essay_text[cursor_interval_to[1]:]
                    )

                # "the cat ran fast", move "ran" to 
                # "ran the cat fast"
                else:

                    essay_text = (
                        # all text preceding move-impacted window
                        essay_text[:cursor_interval_to[0]] + 
                        # add moved block
                        essay_text[cursor_interval_from[0]:cursor_interval_from[1]] +
                        # skip moved block, still within move-impacted window
                        essay_text[cursor_interval_to[0]:cursor_interval_from[0]] + 
                        # all text proceeding move-impacted window
                        essay_text[cursor_interval_from[1]:]
                    )
      
            continue
        

        cursor_position_start_text_change = (
            cursor_position_after_event - len(text_change_log)
            )
        essay_text = (
            essay_text[:cursor_position_start_text_change] 
            + text_change_log
            + essay_text[cursor_position_start_text_change:]
            )
        
    return pd.DataFrame({'id': df['id'].unique(), 'essay': [essay_text]})
