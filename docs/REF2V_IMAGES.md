# Ref2V reference image inputs (November 5, 2026)

MiniMax H3 > R2VA offers nine optional photo upload/drop/path inputs when
Use Multi IMG is off. Filled slots are sent in order as <Picture 1> through
<Picture 9>; empty slots do not create gaps in the model's picture numbering.
Choices persist with the H3 settings. The downloadable Ref2V editor workflow
exposes all nine optional reference-image sockets. Executable worker graphs add
one LoadImage per supplied reference and reuse all nine encoded references for
latent refinement. The console rejects a tenth slot before uploading assets.

Use Multi IMG remains the separate four-percentage-keyframe workflow; FL2V
continues to accept up to four keyframes. Prompt Book is unchanged.
