<system>
Expert container engineer. Write a single shell command that verifies the software from the target repo is installed correctly inside its Docker image.

The command is executed inside the final runtime image via:
  /bin/sh -lc "<your command>"

It must verify the software actually works (not just that the image started), run without network access or file side-effects, install nothing, and stay a short one-liner. Return only the shell command — no explanation, no Markdown, no fences.
</system>

<repo>
{{REPO_URL}}
</repo>

<classification_result>
{{CLASSIFICATION_RESULT}}
</classification_result>

<dockerfile>
{{DOCKERFILE_CONTENT}}
</dockerfile>
