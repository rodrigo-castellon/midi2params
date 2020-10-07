"""
Tiny script meant to generate the HTML tables offline (copy and paste them into the HTML).
"""
import re
import copy

s_str = """<tr>
    <td>##00</td>
    <td><audio controls="controls" src="http://nlp.stanford.edu/data/cdonahue/wavegenie_userstudy/v2/obligation/00.wav">
        Your browser does not support the HTML5 Audio element.
    </audio></td>
    <td><audio controls="controls" src="http://nlp.stanford.edu/data/cdonahue/wavegenie_userstudy/v2/shoulder/00.wav">
        Your browser does not support the HTML5 Audio element.
    </audio></td>
    <td><audio controls="controls" src="http://nlp.stanford.edu/data/cdonahue/wavegenie_userstudy/v2/grandmother/00.wav">
        Your browser does not support the HTML5 Audio element.
    </audio></td>
    <td><audio controls="controls" src="http://nlp.stanford.edu/data/cdonahue/wavegenie_userstudy/v2/art/00.wav">
        Your browser does not support the HTML5 Audio element.
    </audio></td>
    <td><audio controls="controls" src="http://nlp.stanford.edu/data/cdonahue/wavegenie_userstudy/v2/bubble/00.wav">
        Your browser does not support the HTML5 Audio element.
    </audio></td>
</tr>"""

def mod_string(i):
    global s_str

    s = copy.deepcopy(s_str)
    for m in re.finditer('\.wav', s):
        s = s[:m.start() - 2] + str(i).zfill(2) + s[m.start():]
    for m in re.finditer('##', s):
        s = s[:m.start()] + str(i).zfill(2) + s[m.start() + 4:]
    print(s)

if __name__ == '__main__':
    for i in range(19):
        mod_string(i)
