# deploy (systemd user unit)

    mkdir -p ~/.config/systemd/user
    cp deploy/maya-glm-atlas.service ~/.config/systemd/user/
    systemctl --user daemon-reload
    systemctl --user enable --now maya-glm-atlas
    # open http://<tailscale-ip>:8091/  (`tailscale ip -4` for the address)
    # from any device on the tailnet; loopback-only if you drop --bind tailscale

Requirements: modified-maya serving GLM-5.3-Flash (Maya-S-v2-IQ2_XXS pack)
with `/metrics` on 127.0.0.1:8009 (token `sk-local`, or set
`MAYA_METRICS_TOKEN`), python3.12 (stdlib only), and tools/build_atlas.py
having been run at least once (the ExecStartPre re-runs it on every start so
the atlas tracks the GGUF shards + pack).
