  <example index="1">
  <user>
  Analyze this repository for installation classification:
  Repo: https://github.com/tiangolo/fastapi
  Relevant files provided:
  - pyproject.toml: [tool.poetry], python = "^3.8", dependencies include fastapi, uvicorn, pydantic
  - poetry.lock present (locks fastapi, pydantic, uvicorn + dev deps pytest, httpx)
  - No Dockerfile present
  - No .env.example present
  </user>
  <assistant>
  schema_version: "1.1"
  repo: "https://github.com/tiangolo/fastapi"
  branch: "master"
  commit_hash: ""
  analyzed_at: "2026-01-15T10:00:00Z"

  categories:
    programming_language:
      value: ["Python"]

    language_version:
      value:
        Python: ">=3.8"

    build_tool:
      value: ["pip"]

    runtime_environment:
      value: ["native"]

    os_compatibility:
      value: ["Linux", "macOS", "Windows"]

    installation_strategy:
      value: native

    dependencies_packages:
      value:
        runtime: ["fastapi", "pydantic", "uvicorn"]
        dev: ["pytest", "httpx"]
        optional: ["fastapi[all]"]

    system_dependencies:
      value: []

    environment_variables:
      value: []

    config_files:
      value: []

    external_services:
      value: []

    network:
      value:
        ports: []
        outbound_required: false

    permissions:
      value: []

    containerization:
      value:
        dockerfile_present: false
        docker_compose_present: false
        kubernetes_present: false
        devcontainer_present: false

    install_steps:
      value:
        readme: false
        contributing_md: false
        makefile_targets: []
        dedicated_install_doc: false

    setup_steps:
      value:
        - "python3 -m venv venv"
        - "source venv/bin/activate"

    build_steps:
      value:
        - "pip install -e ."

    run_steps:
      value: []

    entrypoints:
      value: []

    post_install_steps:
      value: []

    verification:
      value:
        - "python -c \"import fastapi; print(fastapi.__version__)\""

    idempotency:
      value: true

    known_issues:
      value: []

    source_confidence:
      value:
        programming_language: lockfile
        language_version: heuristic
        build_tool: lockfile
        build_steps: heuristic
        verification: heuristic
  </assistant>
  </example>

  <example index="2">
  <user>
  Analyze this repository for installation classification:
  Repo: https://github.com/calcom/cal.com
  Relevant files provided:
  - package.json: yarn workspaces (turbo), Next.js app; engines node >=18; dependencies: next, react, @prisma/client; devDependencies: typescript, jest
  - yarn.lock present
  - .nvmrc: 18.18.0
  - docker-compose.yml: services: database (postgres:13)
  - .env.example: DATABASE_URL=postgresql://..., NEXTAUTH_SECRET=, NEXTAUTH_URL=http://localhost:3000, CALENDSO_ENCRYPTION_KEY=
  - No Dockerfile present
  </user>
  <assistant>
  schema_version: "1.1"
  repo: "https://github.com/calcom/cal.com"
  branch: "main"
  commit_hash: ""
  analyzed_at: "2026-01-15T10:05:00Z"

  categories:
    programming_language:
      value: ["TypeScript"]

    language_version:
      value:
        Node: ">=18"

    build_tool:
      value: ["yarn"]

    runtime_environment:
      value: ["Node.js"]

    os_compatibility:
      value: ["Linux", "macOS", "Windows"]

    installation_strategy:
      value: native

    dependencies_packages:
      value:
        runtime: ["next", "react", "@prisma/client"]
        dev: ["typescript", "jest"]
        optional: []

    system_dependencies:
      value: []

    environment_variables:
      value:
        - name: "DATABASE_URL"
          required: true
          default: "postgresql://localhost:5432/calendso"
          description: "PostgreSQL connection string"
          sensitive: true
        - name: "NEXTAUTH_SECRET"
          required: true
          default: null
          description: "Secret used to sign NextAuth sessions"
          sensitive: true
        - name: "NEXTAUTH_URL"
          required: true
          default: "http://localhost:3000"
          description: "Canonical URL of the app"
          sensitive: false
        - name: "CALENDSO_ENCRYPTION_KEY"
          required: true
          default: null
          description: "Key for encrypting stored credentials"
          sensitive: true

    config_files:
      value: []

    external_services:
      value:
        - service: "PostgreSQL"
          required: true
          local_option: true

    network:
      value:
        ports: [3000]
        outbound_required: false

    permissions:
      value: []

    containerization:
      value:
        dockerfile_present: false
        docker_compose_present: true
        kubernetes_present: false
        devcontainer_present: false

    install_steps:
      value:
        readme: false
        contributing_md: false
        makefile_targets: []
        dedicated_install_doc: false

    setup_steps:
      value: []

    build_steps:
      value:
        - "yarn install"
        - "yarn build"

    run_steps:
      value:
        - "yarn start"

    entrypoints:
      value:
        - command: "yarn start"
          description: "Start the Next.js application server"
          type: server

    post_install_steps:
      value:
        - "Run database migrations: yarn prisma migrate deploy"

    verification:
      value:
        - "yarn test"

    idempotency:
      value: true

    known_issues:
      value: []

    source_confidence:
      value:
        programming_language: lockfile
        language_version: lockfile
        build_tool: lockfile
        environment_variables: heuristic
        external_services: heuristic
  </assistant>
  </example>
