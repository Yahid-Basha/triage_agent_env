"""
TriageAgent corpus — hardcoded synthetic enterprise data.
Run this once: python data/build_corpus.py
Outputs: kb.json, past_tickets.json, incidents.json, train_tickets.json, eval_tickets.json
"""

import json, pathlib, sys

OUT = pathlib.Path(__file__).parent

# ─── KB ARTICLES (20) ─────────────────────────────────────────────────────────
KB = [
    # NETWORKING
    {
        "article_id": "KB-00001",
        "title": "BGP Peer Session Down — Troubleshooting Guide",
        "domain": "networking",
        "tags": ["bgp", "routing", "peer", "ebgp", "ibgp"],
        "updated": "2025-09-12",
        "body": (
            "BGP peer sessions drop for three primary reasons: TCP connectivity loss, "
            "hold-timer expiry, or configuration mismatch.\n\n"
            "**Step 1 — Verify TCP reachability**\n"
            "Ping the peer address from the router VRF: `ping vrf MGMT <peer-ip> source <local-ip>`. "
            "If ping fails, check ACLs on both ends blocking TCP/179.\n\n"
            "**Step 2 — Check BGP state**\n"
            "`show bgp neighbors <peer-ip>` — look for 'BGP state = Active' (can't connect) vs "
            "'Idle (Admin)' (locally shut). Check 'Hold time' and 'Keepalive interval'.\n\n"
            "**Step 3 — Validate AS and peer config**\n"
            "Confirm `neighbor <ip> remote-as <AS>` matches the peer's local AS. "
            "MD5 password mismatch causes 'BGP notification: hold time expired' logs.\n\n"
            "**Step 4 — Review logs**\n"
            "`debug ip bgp <peer-ip> events` for Cisco IOS. Look for NOTIFICATION messages "
            "indicating cease/hold-timer/open-message-error subtypes.\n\n"
            "**Resolution**\n"
            "Most common fix: correct hold-timer mismatch with `neighbor <ip> timers <keepalive> <hold>` "
            "or clear ACL blocking TCP 179. After fixing, `clear ip bgp <peer-ip> soft` to re-establish "
            "without dropping the session hard.\n\n"
            "**Escalation threshold:** If session stays in Active for >5 min after TCP fix, "
            "escalate to NOC-L2 with `show bgp summary` and `show ip interface brief` output."
        )
    },
    {
        "article_id": "KB-00002",
        "title": "OSPF Neighbor Adjacency Failure Investigation",
        "domain": "networking",
        "tags": ["ospf", "routing", "adjacency", "lsa", "area"],
        "updated": "2025-11-03",
        "body": (
            "OSPF adjacencies fail at specific state transitions. The most common stuck states "
            "are INIT (hello seen but not bidirectional), 2-WAY, and EXSTART/EXCHANGE.\n\n"
            "**Step 1 — Identify stuck state**\n"
            "`show ip ospf neighbor` — if stuck in INIT, the remote router is not receiving your hellos "
            "or not including your router-id in its hello. Check subnet mask and hello/dead intervals match.\n\n"
            "**Step 2 — MTU mismatch (most common EXSTART cause)**\n"
            "OSPF DBD packets use full MTU. If interface MTU differs between peers, EXSTART gets stuck. "
            "Fix: `ip ospf mtu-ignore` on both interfaces, or align MTUs.\n\n"
            "**Step 3 — Area-type mismatch**\n"
            "Stub vs non-stub area mismatch prevents adjacency. Check `show ip ospf` for area flags.\n\n"
            "**Step 4 — Authentication**\n"
            "MD5 key mismatch silently drops hellos. Use `debug ip ospf adj` and look for "
            "'Invalid authentication' messages.\n\n"
            "**Resolution**\n"
            "For MTU: align physical MTU or add `ip ospf mtu-ignore`. "
            "For area mismatch: ensure both routers agree on stub/nssa flags in `area <id> stub` config. "
            "After fix, `clear ip ospf process` is destructive — prefer fixing config and waiting for "
            "dead-timer expiry (default 40s).\n\n"
            "**Related:** KB-00001 (BGP) for comparison on hold-timer behavior."
        )
    },
    {
        "article_id": "KB-00003",
        "title": "Corporate DNS Resolution Failures — Diagnosis and Fix",
        "domain": "networking",
        "tags": ["dns", "resolution", "nslookup", "bind", "forwarder"],
        "updated": "2025-10-22",
        "body": (
            "DNS resolution failures manifest as application connection errors, not network errors. "
            "Always confirm DNS is the issue before escalating network.\n\n"
            "**Step 1 — Isolate DNS vs network**\n"
            "`nslookup <hostname> <dns-server-ip>` directly targeting the corp DNS. If this succeeds "
            "but the application fails, the client is using the wrong DNS server. "
            "If nslookup fails: proceed.\n\n"
            "**Step 2 — Check forwarder chain**\n"
            "Corp DNS servers (10.10.1.53, 10.10.2.53) forward to ISP resolvers for external names. "
            "If external resolution fails but internal succeeds, check forwarder connectivity: "
            "`dig @10.10.1.53 google.com +time=2`. Timeout means forwarder blocked.\n\n"
            "**Step 3 — Zone delegation issues**\n"
            "Internal zones (corp.example.com, svc.example.com) are authoritative on 10.10.1.53. "
            "NXDOMAIN for internal names means either the record is missing or split-DNS routing "
            "is sending the query to external resolvers. Check client DNS server assignment.\n\n"
            "**Step 4 — Flush and retry**\n"
            "Windows: `ipconfig /flushdns`. Linux: `systemd-resolve --flush-caches` or restart "
            "`systemd-resolved`. Negative cache TTL is 300s by default.\n\n"
            "**Resolution**\n"
            "Most common: client has 8.8.8.8 as DNS (DHCP misconfiguration) and can't resolve internal "
            "names. Fix: push correct DNS via DHCP option 6, or manually set to 10.10.1.53."
        )
    },
    {
        "article_id": "KB-00004",
        "title": "F5 LTM Health Monitor Failures — Configuration Guide",
        "domain": "networking",
        "tags": ["f5", "load-balancer", "health-monitor", "pool", "virtual-server"],
        "updated": "2025-08-14",
        "body": (
            "F5 LTM marks pool members red when health monitors fail. This causes 502/503 errors "
            "upstream. Members show as 'down' in the GUI: Local Traffic > Pools > Pool List.\n\n"
            "**Step 1 — Check monitor type vs service**\n"
            "HTTP monitors send GET / HTTP/1.1 and expect a specific response. If the backend "
            "requires Host header or returns 301 to HTTPS, the HTTP monitor will fail. "
            "Use HTTPS monitor with `send: HEAD /health HTTP/1.1\\r\\nHost: myapp.corp.example.com\\r\\n\\r\\n`.\n\n"
            "**Step 2 — Verify receive string**\n"
            "The monitor's `recv` string must appear in the response body. If backend changed the "
            "health check response text, update `recv: 'OK'` accordingly.\n\n"
            "**Step 3 — Connectivity from F5 self-IP**\n"
            "F5 sends health checks from its self-IP (not VIP). Firewalls may block this. "
            "Test: from F5 bash, `curl -H 'Host: myapp.corp.example.com' http://<member-ip>:<port>/health`.\n\n"
            "**Step 4 — Force-up for emergency**\n"
            "iControl REST: `PATCH /mgmt/tm/ltm/pool/~Common~mypool/members/~Common~<ip>:<port>` "
            "with `{\"session\": \"user-enabled\", \"state\": \"user-up\"}`. "
            "Document and create incident ticket — this bypasses health checking.\n\n"
            "**Resolution**\n"
            "Fix monitor send/recv strings, or add firewall rule permitting self-IP health checks. "
            "Prefer fixing root cause over force-up."
        )
    },
    {
        "article_id": "KB-00005",
        "title": "VPN Tunnel Flapping — IPSec IKEv2 Troubleshooting",
        "domain": "networking",
        "tags": ["vpn", "ipsec", "ikev2", "tunnel", "flapping"],
        "updated": "2025-12-01",
        "body": (
            "VPN tunnels that come up and immediately drop (flapping) are almost always caused by "
            "Phase 1 (IKE) or Phase 2 (IPSec) parameter mismatch, or DPD misconfiguration.\n\n"
            "**Step 1 — Capture the NOTIFY payload**\n"
            "`debug crypto ikev2` on Cisco or check VPN gateway logs. Look for NOTIFY messages: "
            "NO_PROPOSAL_CHOSEN means algorithm mismatch. TS_UNACCEPTABLE means traffic selector mismatch.\n\n"
            "**Step 2 — Algorithm comparison**\n"
            "Verify both ends use identical IKE proposal: encryption (AES-256), PRF (SHA-256), "
            "DH group (14 or 19), lifetime (86400s). Even one mismatch causes immediate teardown.\n\n"
            "**Step 3 — Dead Peer Detection (DPD)**\n"
            "Aggressive DPD timers cause tunnels to drop under high latency. "
            "Default: `dpd 30 retry 5`. If WAN latency spikes >20s, increase to `dpd 60 retry 5`.\n\n"
            "**Step 4 — NAT-T**\n"
            "If one end is behind NAT, NAT-T must be enabled on both sides (UDP 4500). "
            "Check: `show crypto ikev2 sa` — if NAT-T flag is missing on one side, tunnel fails Phase 1.\n\n"
            "**Resolution**\n"
            "Align IKE proposal on both ends. If DPD: increase timers. "
            "If NAT-T: enable `crypto ikev2 nat-keepalive 30` on both sides."
        )
    },
    {
        "article_id": "KB-00006",
        "title": "DHCP Scope Exhaustion — Emergency Recovery",
        "domain": "networking",
        "tags": ["dhcp", "ip-address", "scope", "lease", "exhaustion"],
        "updated": "2025-07-28",
        "body": (
            "DHCP scope exhaustion prevents new devices from getting IP addresses. "
            "Clients fall back to APIPA (169.254.x.x) and cannot communicate.\n\n"
            "**Immediate mitigation (< 5 min)**\n"
            "1. `show ip dhcp pool` to confirm utilization. If >95%, proceed.\n"
            "2. `clear ip dhcp binding *` — WARNING: this forces all clients to renew. "
            "   Do only in maintenance window or if stale leases are the cause.\n"
            "3. Alternative: `clear ip dhcp binding <specific-ip>` for targeted stale entries.\n\n"
            "**Identify stale leases**\n"
            "`show ip dhcp binding | include Expiry` — entries with expiry >7 days in the future "
            "are likely stale (device removed but lease not expired). Cross-reference with ARP table: "
            "`show ip arp | include <subnet>`. Entries with 'Incomplete' are orphaned.\n\n"
            "**Permanent fix options**\n"
            "A) Reduce lease time from 8d to 1d: `ip dhcp pool CORP_WIFI / lease 1`\n"
            "B) Expand scope: if /24 is full, supernet to /23 or add a new secondary pool\n"
            "C) Enable DHCP snooping to prevent rogue DHCP servers consuming addresses\n\n"
            "**Prevention**\n"
            "Alert at 80% utilization. SNMP OID: 1.3.6.1.4.1.9.9.243.1.3.1.10 for Cisco DHCP pool usage."
        )
    },
    {
        "article_id": "KB-00007",
        "title": "Interface CRC Error Investigation and Remediation",
        "domain": "networking",
        "tags": ["crc", "interface", "errors", "duplex", "cabling"],
        "updated": "2025-10-05",
        "body": (
            "CRC errors indicate data corruption at layer 1/2. High CRC rates cause retransmissions, "
            "TCP slowdown, and packet loss even when interface shows 'up/up'.\n\n"
            "**Severity thresholds**\n"
            "< 0.01% of input packets: acceptable. 0.01–1%: investigate. >1%: urgent remediation.\n\n"
            "**Step 1 — Baseline counters**\n"
            "`show interface <int> | include CRC|input|output` — note the count and timestamp. "
            "Wait 5 min, recheck. Divide delta CRCs by delta input packets for rate.\n\n"
            "**Step 2 — Check duplex mismatch (most common cause)**\n"
            "`show interface <int> | include duplex` — if one side is full-duplex and the other is "
            "half or auto, CRCs accumulate on the full-duplex side. Fix: hard-set speed and duplex "
            "on both ends: `speed 1000 / duplex full`.\n\n"
            "**Step 3 — Physical layer check**\n"
            "For SFP: `show interfaces <int> transceiver detail` — Rx power below -20 dBm is "
            "marginal. Reseat SFP. For copper: check cable length (Cat5e max 100m at 1G).\n\n"
            "**Step 4 — VLAN/trunk config**\n"
            "Native VLAN mismatch on trunks causes FCS errors, not CRC, but presents similarly. "
            "`show interfaces trunk` to verify.\n\n"
            "**Resolution**\n"
            "Fix duplex mismatch first (covers 70% of cases). If still elevated after duplex fix, "
            "replace cable or SFP. Escalate to hardware team if physical replacement needed."
        )
    },

    # IDENTITY
    {
        "article_id": "KB-00008",
        "title": "Active Directory Authentication Failures — Troubleshooting",
        "domain": "identity",
        "tags": ["active-directory", "ldap", "kerberos", "authentication", "ad"],
        "updated": "2025-09-30",
        "body": (
            "AD authentication failures present as 'Invalid credentials' or 'Account locked' "
            "errors. The Windows Event Log is the primary diagnostic tool.\n\n"
            "**Step 1 — Check lockout status**\n"
            "`Get-ADUser <username> -Properties LockedOut,BadLogonCount,LastBadPasswordAttempt` "
            "in PowerShell. If LockedOut=True, unlock: `Unlock-ADAccount -Identity <username>`.\n\n"
            "**Step 2 — Find lockout source**\n"
            "Event ID 4740 on the PDC Emulator shows the source computer locking the account. "
            "Use Microsoft's free Account Lockout Status (LockoutStatus.exe) tool. "
            "Common source: mapped drives or services using cached old password.\n\n"
            "**Step 3 — Password expiry**\n"
            "`(Get-ADUser <username> -Properties PasswordExpired).PasswordExpired` — if True, "
            "user must reset. Admins cannot see the password; reset via `Set-ADAccountPassword`.\n\n"
            "**Step 4 — Kerberos ticket issues**\n"
            "Event ID 4771 (pre-auth failed) with error code 0x18 = wrong password. "
            "0x12 = account disabled. 0x25 = clock skew >5 min (Kerberos tolerance). "
            "Fix clock skew: `w32tm /resync /force` on the affected machine.\n\n"
            "**Step 5 — LDAP bind test**\n"
            "`ldp.exe` (Windows) or `ldapsearch -H ldap://corp-dc1.corp.example.com -D 'user@corp.example.com' "
            "-W -b 'DC=corp,DC=example,DC=com' '(sAMAccountName=<username>)'` to test bind directly.\n\n"
            "**Resolution**\n"
            "Unlock account, fix password, sync clock. If recurring lockouts, identify source via LockoutStatus."
        )
    },
    {
        "article_id": "KB-00009",
        "title": "Okta SCIM 2.0 Provisioning Setup and Troubleshooting",
        "domain": "identity",
        "tags": ["okta", "scim", "provisioning", "idp", "sso"],
        "updated": "2025-11-15",
        "body": (
            "SCIM 2.0 provisioning syncs user lifecycle (create/update/deactivate) from Okta "
            "to downstream applications. Misconfiguration causes silent provisioning failures.\n\n"
            "**Setup (new integration)**\n"
            "1. In Okta Admin: Applications > App > Provisioning tab > Enable SCIM provisioning.\n"
            "2. SCIM connector base URL: `https://<your-app>/scim/v2/`\n"
            "3. Auth: select HTTP Header. Generate a Bearer token in the target app and paste it.\n"
            "4. Test connector: Okta sends a GET /scim/v2/Users request. Status 200 = working.\n"
            "5. Enable: Push New Users, Push Profile Updates, Push Groups, Deactivate Users.\n\n"
            "**Attribute mapping (critical)**\n"
            "Required SCIM attributes: `userName` (maps to email), `name.givenName`, `name.familyName`. "
            "Optional but common: `phoneNumbers[0].value`, `title`, `department`.\n\n"
            "**Common failures**\n"
            "- 401: Bearer token expired or wrong. Regenerate in the target app.\n"
            "- 404 on /scim/v2/Users: SCIM endpoint not enabled in target app config.\n"
            "- User created in app but profile not updated: Push Profile Updates not enabled.\n"
            "- User not deactivated: Deactivate Users toggle must be ON; some apps use `active: false`.\n\n"
            "**Debugging**\n"
            "Okta System Log: filter by `event_type eq \"application.provision.user.push\"`. "
            "Error messages include the SCIM response body from the downstream app.\n\n"
            "**Resolution**\n"
            "Regenerate Bearer token if 401. Enable SCIM in target app if 404. "
            "Check attribute mapping for profile-sync issues."
        )
    },
    {
        "article_id": "KB-00010",
        "title": "SAML 2.0 SSO Configuration — IdP and SP Setup",
        "domain": "identity",
        "tags": ["saml", "sso", "idp", "sp", "federation"],
        "updated": "2025-08-22",
        "body": (
            "SAML 2.0 SSO errors are cryptic. The two most common failures are clock skew "
            "and assertion attribute mismatch.\n\n"
            "**Core configuration checklist**\n"
            "IdP (Okta/ADFS) side:\n"
            "- SP Entity ID (Audience): must exactly match what the SP expects. Case-sensitive.\n"
            "- ACS URL: the SP's assertion consumer service URL. Usually `/saml/acs` or `/sso/saml`.\n"
            "- NameID format: `emailAddress` is most common. Some SPs require `unspecified`.\n"
            "SP side:\n"
            "- IdP SSO URL: copy exactly from IdP metadata XML.\n"
            "- IdP Signing Certificate: download X.509 cert from IdP metadata. Renew annually.\n\n"
            "**Troubleshooting with SAML Tracer (Chrome extension)**\n"
            "1. Install SAML Tracer, start recording.\n"
            "2. Attempt SSO login.\n"
            "3. Look for the POST to the ACS URL. Decode the SAMLResponse Base64 payload.\n"
            "4. Check `<Conditions NotBefore NotOnOrAfter>` — if expired, clock skew is the cause.\n\n"
            "**Common errors**\n"
            "- 'Audiences does not match': Entity ID mismatch between IdP and SP.\n"
            "- 'InResponseTo mismatch': SP reused an old AuthnRequest ID. Clear SP session.\n"
            "- 'Signature verification failed': IdP certificate changed, update SP trust.\n\n"
            "**Resolution**\n"
            "Fix Entity ID case mismatch, sync NTP (skew tolerance is 300s), update signing cert."
        )
    },
    {
        "article_id": "KB-00011",
        "title": "MFA Reset and Account Unlock Procedure",
        "domain": "identity",
        "tags": ["mfa", "totp", "reset", "locked", "authenticator"],
        "updated": "2025-10-18",
        "body": (
            "Users locked out of MFA cannot self-service. This procedure is for IT admins only. "
            "Always verify user identity via video call or badge scan before resetting MFA.\n\n"
            "**Okta MFA reset**\n"
            "1. Okta Admin Console > Directory > People > Search user.\n"
            "2. Click user > More Actions > Reset Multifactor.\n"
            "3. Confirm reset. User receives an activation email and must re-enroll.\n"
            "4. Do NOT reset if the user has active sessions that could be hijacked. "
            "   Check 'Current Sessions' and terminate all before reset.\n\n"
            "**Google Workspace MFA reset**\n"
            "`gam update user <email> is2svEnrolled false` — then notify user to re-enroll "
            "at myaccount.google.com/signinoptions/two-step-verification.\n\n"
            "**Azure AD / Entra MFA reset**\n"
            "Azure Portal > Users > Select user > Authentication methods > Require re-register MFA.\n"
            "Or via PowerShell: `Set-MgUserAuthenticationRequirement -UserId <objectId> "
            "-PerUserMfaState Disabled`\n\n"
            "**Backup verification codes**\n"
            "If the user has backup codes stored securely, they can self-recover. "
            "Ask before resetting — avoids the re-enrollment friction.\n\n"
            "**After reset**\n"
            "Log the reset in the IT ticket system with: user name, time, admin who performed reset, "
            "verification method used. Required for compliance audit."
        )
    },
    {
        "article_id": "KB-00012",
        "title": "Service Account Password Rotation Procedure",
        "domain": "identity",
        "tags": ["service-account", "password", "rotation", "ad", "automation"],
        "updated": "2025-12-10",
        "body": (
            "Service account passwords must be rotated every 90 days per security policy. "
            "Uncoordinated rotation breaks dependent services. Follow this procedure.\n\n"
            "**Step 1 — Impact assessment**\n"
            "Before rotating, identify all dependencies:\n"
            "`Get-ADUser <svc-account> -Properties ServicePrincipalNames,Description` — check Description "
            "for a list of dependent services. Also search config files: "
            "`grep -r '<service-account-name>' /etc/`\n\n"
            "**Step 2 — Update in PAM vault first**\n"
            "Update the new password in CyberArk / HashiCorp Vault BEFORE changing in AD. "
            "This ensures downstream services can retrieve it atomically.\n\n"
            "**Step 3 — Rotate in AD**\n"
            "`Set-ADAccountPassword -Identity <svc-account> -NewPassword (ConvertTo-SecureString "
            "'<new-password>' -AsPlainText -Force) -Reset`\n\n"
            "**Step 4 — Update dependent services (in order)**\n"
            "1. Windows services: `sc config <service> password= <new-password>` then restart service.\n"
            "2. IIS Application Pools: IIS Manager > App Pools > Advanced Settings > Identity.\n"
            "3. Scheduled tasks: Task Scheduler > Properties > General > Change User.\n"
            "4. JDBC/connection strings: update config files, restart application.\n\n"
            "**Step 5 — Verify**\n"
            "Watch service logs for authentication errors for 15 min after rotation. "
            "If any service fails, roll back to old password immediately, then re-coordinate.\n\n"
            "**For gMSA (Group Managed Service Accounts):** rotation is automatic every 30 days. "
            "No manual rotation needed. Prefer gMSA over traditional service accounts."
        )
    },
    {
        "article_id": "KB-00013",
        "title": "API Token Rotation and Revocation",
        "domain": "identity",
        "tags": ["api-token", "rotation", "revocation", "credentials", "security"],
        "updated": "2025-09-05",
        "body": (
            "API tokens compromised or expired must be revoked and rotated with zero downtime. "
            "This requires a brief window where both old and new tokens are valid.\n\n"
            "**Zero-downtime rotation pattern**\n"
            "1. Generate NEW token while OLD token is still valid.\n"
            "2. Update all consumers of the token (one by one or via rolling deployment).\n"
            "3. Verify all consumers are using the new token (check access logs).\n"
            "4. Revoke the OLD token.\n\n"
            "**GitHub Personal Access Tokens (PATs)**\n"
            "Settings > Developer Settings > Personal access tokens > Fine-grained tokens. "
            "Set expiry to 90 days max per policy. Rotation reminders should be set at -14 days.\n\n"
            "**Jenkins API Tokens**\n"
            "User > Configure > API Token > Add new token. Update Jenkinsfile credentials "
            "binding before removing old token from the user account.\n\n"
            "**Generic REST API tokens**\n"
            "Most platforms: POST /api/v1/tokens to create, DELETE /api/v1/tokens/{id} to revoke. "
            "Store tokens in Vault: `vault kv put secret/svc/<service>/api-token value=<token>`.\n\n"
            "**Emergency revocation (token compromised)**\n"
            "Revoke first, update consumers second. Accept a brief outage rather than a prolonged "
            "security exposure. File a P1 incident and involve the security team.\n\n"
            "**Audit**\n"
            "After rotation, verify old token returns 401 from an external test: "
            "`curl -H 'Authorization: Bearer <old-token>' https://api.example.com/health`"
        )
    },
    {
        "article_id": "KB-00014",
        "title": "TLS Certificate Renewal for Internal Services",
        "domain": "identity",
        "tags": ["tls", "ssl", "certificate", "renewal", "expiry", "pki"],
        "updated": "2025-11-28",
        "body": (
            "Expired TLS certificates cause hard failures in strict clients (curl, browsers with HSTS). "
            "Internal services use the corp CA. Renewal must happen ≥14 days before expiry.\n\n"
            "**Check expiry**\n"
            "`openssl s_client -connect <host>:<port> -servername <host> </dev/null 2>/dev/null | "
            "openssl x509 -noout -dates`\n"
            "Or: `echo | openssl s_client -connect <host>:443 2>/dev/null | openssl x509 -noout "
            "-checkend 1209600` — returns non-zero if expiring within 14 days.\n\n"
            "**Request renewal from internal CA**\n"
            "1. Generate CSR: `openssl req -new -newkey rsa:2048 -nodes -keyout <service>.key "
            "   -out <service>.csr -subj '/CN=<fqdn>/O=Corp/C=US'`\n"
            "2. Add SANs: create config file with `subjectAltName = DNS:<fqdn>,DNS:<alias>,IP:<ip>`\n"
            "3. Submit CSR to IT-PKI team via ServiceNow ticket category 'Certificate > Internal PKI'.\n"
            "4. SLA: 2 business days for standard, 4 hours for P1 (expired cert causing outage).\n\n"
            "**Install renewed certificate**\n"
            "Nginx: update `ssl_certificate` and `ssl_certificate_key` paths, `nginx -t && nginx -s reload`\n"
            "Apache: update `SSLCertificateFile`, `apachectl configtest && apachectl graceful`\n"
            "Java keystore: `keytool -importcert -alias <alias> -file <cert.pem> -keystore <keystore.jks>`\n\n"
            "**Verify**\n"
            "`curl -v https://<host>` — check 'expire date' in TLS handshake output."
        )
    },

    # APP SUPPORT
    {
        "article_id": "KB-00015",
        "title": "JVM Out of Memory Error — Heap Dump Analysis",
        "domain": "app_support",
        "tags": ["jvm", "oom", "heap", "memory", "spring-boot"],
        "updated": "2025-10-12",
        "body": (
            "JVM OOM crashes are either heap exhaustion (java.lang.OutOfMemoryError: Java heap space) "
            "or metaspace exhaustion (java.lang.OutOfMemoryError: Metaspace). "
            "Heap dumps are mandatory for root cause analysis.\n\n"
            "**Enable automatic heap dump on OOM**\n"
            "Add JVM flags: `-XX:+HeapDumpOnOutOfMemoryError -XX:HeapDumpPath=/var/log/heapdumps/`\n"
            "Also add: `-XX:+ExitOnOutOfMemoryError` to force a clean restart (avoids zombie state).\n\n"
            "**Capture heap dump on running process**\n"
            "`jmap -dump:format=b,file=/tmp/heap_$(date +%s).hprof <pid>`\n"
            "Note: jmap freezes the JVM for the duration. Prefer OOM trigger for production.\n\n"
            "**Analyze with Eclipse MAT**\n"
            "1. Open heap dump: File > Open Heap Dump.\n"
            "2. Run 'Leak Suspects' report.\n"
            "3. Look for the 'Problem Suspect' with the largest retained heap.\n"
            "4. Drill into the dominator tree — large byte[] or char[] arrays usually indicate "
            "   String interning or cache bloat.\n\n"
            "**Common causes**\n"
            "- Unbounded cache: Guava Cache or Caffeine without `maximumSize` configured.\n"
            "- Session object bloat: HttpSession storing large objects without TTL.\n"
            "- Memory leak in third-party library: check version changelogs.\n\n"
            "**Quick mitigation**\n"
            "Increase heap: `-Xmx4g` (from default 1-2g). This buys time for root cause analysis "
            "but does not fix the underlying leak."
        )
    },
    {
        "article_id": "KB-00016",
        "title": "Database Connection Pool Exhaustion — Diagnosis and Recovery",
        "domain": "app_support",
        "tags": ["database", "connection-pool", "hikari", "timeout", "jdbc"],
        "updated": "2025-08-30",
        "body": (
            "Connection pool exhaustion manifests as `HikariPool-1 - Connection is not available, "
            "request timed out after 30000ms` in application logs. Services become degraded but "
            "the database itself may be healthy.\n\n"
            "**Step 1 — Confirm pool exhaustion**\n"
            "Check Prometheus/Grafana metric `hikaricp_connections_active`. If it equals "
            "`hikaricp_connections_max`, the pool is saturated. Also check `hikaricp_connections_pending`.\n\n"
            "**Step 2 — Find connection holders**\n"
            "In the application thread dump: `kill -3 <pid>` (Linux). Search for threads in "
            "`RUNNABLE` or `WAITING` state with JDBC/database stack frames. "
            "Long-running queries or unclosed ResultSets are common causes.\n\n"
            "**Step 3 — Check database side**\n"
            "`SELECT count(*), state FROM pg_stat_activity GROUP BY state;` (PostgreSQL) or "
            "`SHOW PROCESSLIST;` (MySQL). High 'idle' connection count = pool is holding connections "
            "not in use. High 'active' = queries running long.\n\n"
            "**Immediate recovery**\n"
            "Restart the application service to flush the pool. Monitor that connections return "
            "to normal levels (< 50% of max pool size) after restart.\n\n"
            "**Permanent fix**\n"
            "1. HikariCP config: `maximumPoolSize=20` (increase if needed), `connectionTimeout=30000`, "
            "   `idleTimeout=600000`, `maxLifetime=1800000`, `leakDetectionThreshold=60000`.\n"
            "2. Ensure all JDBC code uses try-with-resources to close connections.\n"
            "3. Set query timeout: `spring.datasource.hikari.initializationFailTimeout=1`.\n"
            "**Related:** INC-0003 covers the March payment service outage caused by this issue."
        )
    },
    {
        "article_id": "KB-00017",
        "title": "Kubernetes Pod CrashLoopBackOff — Root Cause Diagnosis",
        "domain": "app_support",
        "tags": ["kubernetes", "k8s", "crashloop", "pod", "container"],
        "updated": "2025-09-20",
        "body": (
            "CrashLoopBackOff means the container started, exited non-zero, and Kubernetes "
            "is retrying with exponential backoff (10s, 20s, 40s, ... max 5m).\n\n"
            "**Step 1 — Get the last crash logs**\n"
            "`kubectl logs <pod> --previous` — this shows the logs from the PREVIOUS container "
            "run, not the current (which may be in backoff). Critical: the current run logs are "
            "often empty if the container crashes in <1s.\n\n"
            "**Step 2 — Check events**\n"
            "`kubectl describe pod <pod>` — look at Events section. "
            "Common messages: 'OOMKilled' (memory limit hit), 'Error' (non-zero exit), "
            "'CreateContainerConfigError' (bad env var or secret reference).\n\n"
            "**Step 3 — Common root causes**\n"
            "- **OOMKilled**: `resources.limits.memory` too low. Increase or fix memory leak.\n"
            "- **Missing secret**: `kubectl get secret <name>` — if not found, the secret doesn't exist "
            "  in that namespace.\n"
            "- **Config error**: bad YAML injected via ConfigMap. Check `kubectl get cm <name> -o yaml`.\n"
            "- **Liveness probe too aggressive**: if probe fails before app is ready, Kubernetes kills it. "
            "  Increase `initialDelaySeconds` to give app time to start.\n\n"
            "**Step 4 — Debug mode**\n"
            "Temporarily set `command: ['sh', '-c', 'sleep 3600']` in the container spec to keep it "
            "alive without the app. Then `kubectl exec -it <pod> -- sh` to inspect the environment.\n\n"
            "**Resolution**\n"
            "Fix root cause per above. Remove debug command. Verify logs show healthy startup."
        )
    },
    {
        "article_id": "KB-00018",
        "title": "API Gateway 504 Timeout Troubleshooting",
        "domain": "app_support",
        "tags": ["api-gateway", "504", "timeout", "latency", "kong", "nginx"],
        "updated": "2025-10-30",
        "body": (
            "504 Gateway Timeout means the upstream service did not respond within the gateway's "
            "configured timeout. This is distinct from 502 (upstream returned an error) or "
            "503 (upstream unreachable).\n\n"
            "**Step 1 — Identify the slow upstream**\n"
            "API gateway access logs include `upstream_response_time` (Kong/Nginx). "
            "Filter for 504s: `grep 504 /var/log/kong/access.log | awk '{print $7, $9}' | sort -n`.\n\n"
            "**Step 2 — Test upstream directly**\n"
            "Bypass the gateway and hit the upstream service: "
            "`curl -w '%{time_total}' -o /dev/null http://<upstream-host>:<port>/endpoint`. "
            "If this also times out, the issue is in the upstream service, not the gateway.\n\n"
            "**Step 3 — Common upstream causes**\n"
            "- DB query regression: explain plan on slow queries, check for missing index.\n"
            "- Downstream dependency slow: the upstream service is waiting on its own dependency.\n"
            "- Thread pool exhaustion: service is handling requests but has no threads for new work.\n\n"
            "**Step 4 — Adjust timeout configuration**\n"
            "Kong: `proxy_read_timeout 60000` (in milliseconds) per route. "
            "Nginx: `proxy_read_timeout 60s` in location block. "
            "Note: increasing timeouts masks the real problem. Investigate before increasing.\n\n"
            "**Step 5 — Circuit breaker**\n"
            "If upstream is intermittently slow, configure circuit breaker in Kong "
            "(circuit-breaker plugin) to fail-fast and return 503 instead of hanging until timeout.\n\n"
            "**Resolution**\n"
            "Fix slow queries, add timeouts in upstream service to its own dependencies, "
            "or increase pool size. Only increase gateway timeout as a last resort."
        )
    },
    {
        "article_id": "KB-00019",
        "title": "Redis Cache Eviction and Key Expiry Issues",
        "domain": "app_support",
        "tags": ["redis", "cache", "eviction", "ttl", "memory"],
        "updated": "2025-07-14",
        "body": (
            "Redis eviction or unexpected key expiry causes cache misses to spike, "
            "increasing database load and response latency.\n\n"
            "**Check eviction policy**\n"
            "`redis-cli CONFIG GET maxmemory-policy` — default is `noeviction` (Redis returns OOM error) "
            "or `allkeys-lru` (evicts least-recently-used). For a cache use case, `allkeys-lru` is correct.\n\n"
            "**Check memory pressure**\n"
            "`redis-cli INFO memory` — compare `used_memory_rss` to `maxmemory`. "
            "If >90%, eviction is active. `redis-cli INFO stats | grep evicted_keys` shows total evicted.\n\n"
            "**Find keys without TTL**\n"
            "`redis-cli --scan --pattern '*' | xargs -L 1 redis-cli ttl | grep -c '^-1'` — "
            "prints count of keys with no expiry. These fill memory indefinitely.\n\n"
            "**Common misconfiguration: no TTL on session keys**\n"
            "Spring Session with Redis stores sessions forever by default unless `spring.session.timeout` "
            "is set. Add `spring.session.redis.cleanup-cron=0 * * * * *` to enable cleanup.\n\n"
            "**Fix for immediate relief**\n"
            "1. Scale up Redis memory: `redis-cli CONFIG SET maxmemory 4gb`.\n"
            "2. Or flush stale cache: `redis-cli FLUSHDB` — WARNING: causes cold cache. "
            "   Coordinate with application team.\n\n"
            "**Permanent fix**\n"
            "Add TTL to all cache keys at write time: `SETEX key <ttl-seconds> value`. "
            "For Spring Cache: `@Cacheable(cacheNames = 'products', cacheManager = ttlCacheManager(300))`."
        )
    },
    {
        "article_id": "KB-00020",
        "title": "Application Deployment Rollback Procedure",
        "domain": "app_support",
        "tags": ["deployment", "rollback", "kubernetes", "helm", "canary"],
        "updated": "2025-12-05",
        "body": (
            "Rollback must be executed within 15 minutes of a bad deployment per SLA. "
            "This procedure covers Kubernetes/Helm deployments (most services) and legacy VM deployments.\n\n"
            "**Kubernetes/Helm rollback (fastest)**\n"
            "`helm rollback <release-name> 0` — rolls back to the previous revision. "
            "Or: `kubectl rollout undo deployment/<name>` for non-Helm deployments.\n"
            "Check history: `helm history <release-name>` or `kubectl rollout history deployment/<name>`.\n\n"
            "**Verify rollback**\n"
            "`kubectl rollout status deployment/<name> --timeout=120s` — waits up to 2 min for pods "
            "to be ready. Check `kubectl get pods` for pod status and age.\n\n"
            "**VM-based rollback**\n"
            "Services on VMs use symlink-based deployment: `ln -sfn /opt/app/releases/<prev-version> "
            "/opt/app/current` then `systemctl restart <service>`.\n"
            "Active releases: `ls -lt /opt/app/releases/` — newest first.\n\n"
            "**Database migration rollback**\n"
            "If the bad deployment ran a DB migration, rollback may not be automatic. "
            "Check `/db/migrations/` for a corresponding `V<n>__down.sql` file. "
            "If no down migration exists, escalate to DBA team — a data restore from backup may be needed.\n\n"
            "**Communication**\n"
            "Post to #incidents Slack channel: deployment rolled back, reason, impact window. "
            "Update incident ticket. Schedule post-mortem within 48h.\n\n"
            "**Related:** INC-0007 for a rollback that required DB intervention."
        )
    },
]

# ─── PAST TICKETS (15) ────────────────────────────────────────────────────────
PAST_TICKETS = [
    # NETWORKING
    {
        "ticket_id": "TKT-100001",
        "title": "BGP session to AS65002 dropping every 2 hours",
        "domain": "networking",
        "description": "MPLS router PE01 loses BGP session to provider edge AS65002 at irregular intervals, approximately every 2 hours. Traffic reroutes via backup path but latency increases by 40ms during the flap. Events started after firewall policy update on 2025-11-01.",
        "comments": [
            {"author": "alice.chen", "text": "Ran `show bgp neighbors 10.0.0.1` — session in Active state. TCP 179 ping OK from router but not from firewall ACL check."},
            {"author": "bob.kumar", "text": "Found it. Firewall policy was updated to block established TCP sessions older than 7200s (2 hours). BGP hold timer is 90s but TCP connection was getting killed at the firewall. Added ACL entry permitting established sessions on TCP 179."},
        ],
        "resolution": "Added a stateful firewall rule to permit established TCP/179 sessions without timeout. BGP session has been stable for 48 hours post-fix.",
        "tags": ["bgp", "firewall", "acl", "tcp179"],
        "status": "Resolved",
        "closed_date": "2025-11-03"
    },
    {
        "ticket_id": "TKT-100002",
        "title": "Wifi users on VLAN 30 cannot resolve internal hostnames",
        "domain": "networking",
        "description": "Corporate wifi users (VLAN 30, 10.30.0.0/22) cannot resolve internal hostnames like corp-mail.corp.example.com. External DNS works fine. Issue started today. Office users on VLAN 10 are not affected.",
        "comments": [
            {"author": "net.ops", "text": "Checked DHCP options for VLAN 30. DNS server option 6 was overwritten during wireless controller upgrade — set to 8.8.8.8 instead of 10.10.1.53."},
        ],
        "resolution": "Updated DHCP option 6 on VLAN 30 scope to 10.10.1.53, 10.10.2.53. Clients auto-renewed. Verified with `nslookup corp-mail.corp.example.com 10.10.1.53` succeeding from wifi clients.",
        "tags": ["dns", "dhcp", "wifi", "vlan"],
        "status": "Resolved",
        "closed_date": "2025-10-15"
    },
    {
        "ticket_id": "TKT-100003",
        "title": "High CRC errors on uplink between SW-CORE-01 and SW-DIST-02",
        "domain": "networking",
        "description": "Network monitoring showing 2.3% CRC error rate on TenGigabitEthernet1/0/1 of SW-CORE-01 toward SW-DIST-02. Started after patch work in the datacenter over the weekend. No connectivity loss yet but trending up.",
        "comments": [
            {"author": "dc.team", "text": "Inspected cabling — patch cable in rack was partially unseated after the weekend work. Also noticed one end was Cat5 (not Cat5e) which is below spec for 10GbE over the 12m run."},
            {"author": "alice.chen", "text": "Replaced with Cat6a patch cable, reseated both ends. Cleared interface counters. CRC rate dropped to 0 after replacement."},
        ],
        "resolution": "Replaced substandard Cat5 patch cable with Cat6a. CRC errors resolved completely.",
        "tags": ["crc", "cabling", "10gbe", "interface"],
        "status": "Resolved",
        "closed_date": "2025-09-29"
    },
    {
        "ticket_id": "TKT-100004",
        "title": "DHCP pool 192.168.50.0/24 at 100% utilization — new devices can't connect",
        "domain": "networking",
        "description": "Security team added 45 new IoT sensors to conference rooms on VLAN 50. Now the /24 pool is exhausted and new devices fall back to APIPA. Urgent — conference room AV systems affected.",
        "comments": [
            {"author": "net.ops", "text": "Identified 38 stale leases from decommissioned devices. Used `show ip dhcp binding` cross-referenced against ARP table. Cleared 38 specific stale bindings."},
            {"author": "net.ops", "text": "Longer term fix: expanded scope to 192.168.50.0/23 to accommodate growth. Also set lease time from 8 days to 1 day for this VLAN."},
        ],
        "resolution": "Cleared 38 stale DHCP bindings, expanded scope to /23, reduced lease time to 1 day. Capacity now at 23%.",
        "tags": ["dhcp", "pool", "exhaustion", "iot"],
        "status": "Resolved",
        "closed_date": "2025-11-20"
    },
    {
        "ticket_id": "TKT-100005",
        "title": "Site-to-site VPN to branch office Bangkok flapping",
        "domain": "networking",
        "description": "IKEv2 tunnel to Bangkok branch (peer 203.0.113.45) keeps resetting every 15-30 minutes. Business says file transfers to Bangkok are failing mid-transfer. London and Singapore tunnels are stable.",
        "comments": [
            {"author": "vpn.team", "text": "Compared IKE proposals. HQ has DH group 14, Bangkok branch router was configured with DH group 2 (1024-bit). NOTIFY messages show NO_PROPOSAL_CHOSEN on every renegotiation attempt. Bangkok router is running older firmware with only group 2."},
            {"author": "vpn.team", "text": "Temporarily configured HQ to accept group 2 for this peer. Opened ticket with Bangkok IT to upgrade firmware to support group 14+."},
        ],
        "resolution": "Temporary: aligned DH group to 2 on HQ peer config. Permanent: Bangkok router firmware updated to 15.2, DH group 14 configured. Tunnel stable for 5 days.",
        "tags": ["vpn", "ipsec", "ikev2", "dh-group"],
        "status": "Resolved",
        "closed_date": "2025-10-28"
    },

    # IDENTITY
    {
        "ticket_id": "TKT-100006",
        "title": "svc-deploy-prod account locked causing CI/CD failures",
        "domain": "identity",
        "description": "All production deployments failing with authentication error. Jenkins pipelines exit with 'LDAP bind failed for svc-deploy-prod'. Account is used by Jenkins, Ansible, and the Kubernetes cluster for image pulls. Issue started 14:32 UTC.",
        "comments": [
            {"author": "iam.team", "text": "Account is locked — 8 bad password attempts. Source: Kubernetes controller-manager node k8s-ctrl-01 still using old password from 3 weeks ago. Password was rotated in Jenkins but not in the K8s secret."},
            {"author": "sre.team", "text": "Updated K8s imagePullSecret: `kubectl create secret docker-registry regcred --docker-password=<new-pw> -n prod --dry-run=client -o yaml | kubectl apply -f -`. Account unlocked. All pods recovering."},
        ],
        "resolution": "Unlocked svc-deploy-prod. Updated Kubernetes imagePullSecret in prod namespace. Root cause: incomplete rotation — K8s secret was missed during the 3-week-ago password rotation.",
        "tags": ["service-account", "locked", "kubernetes", "jenkins"],
        "status": "Resolved",
        "closed_date": "2025-10-14"
    },
    {
        "ticket_id": "TKT-100007",
        "title": "Okta SCIM provisioning not deactivating offboarded users in Salesforce",
        "domain": "identity",
        "description": "HR reports that 12 offboarded employees still have active Salesforce accounts 3 weeks after offboarding. Okta shows users as deactivated. Security audit flagged this as a compliance gap.",
        "comments": [
            {"author": "iam.team", "text": "Investigated Okta System Log. Provisioning push events show 'User deactivated in Okta' but no 'Deprovisioning push' events to Salesforce. The Deactivate Users toggle in Salesforce app provisioning settings was OFF."},
            {"author": "iam.team", "text": "Enabled Deactivate Users toggle in Okta > Salesforce app > Provisioning. Triggered manual deprovisioning for all 12 affected users. Running test with a test user: deactivation in Okta propagated to Salesforce within 2 minutes."},
        ],
        "resolution": "Enabled 'Deactivate Users' in Okta SCIM provisioning for the Salesforce integration. 12 orphaned accounts deactivated. Added monitoring alert for provisioning failures.",
        "tags": ["okta", "scim", "salesforce", "offboarding", "provisioning"],
        "status": "Resolved",
        "closed_date": "2025-09-12"
    },
    {
        "ticket_id": "TKT-100008",
        "title": "User reports SAML SSO login loop on internal wiki",
        "domain": "identity",
        "description": "Three users report they get stuck in a redirect loop when logging into wiki.corp.example.com. They log in, get sent to Okta, authenticate, get sent back to wiki, and then immediately get redirected to Okta again. Other users are unaffected.",
        "comments": [
            {"author": "app.support", "text": "Using SAML Tracer, found the SAMLResponse contains correct attributes. The issue is wiki is generating a new AuthnRequest ID on every response, treating it as a new session. Suspect stale cookies."},
            {"author": "app.support", "text": "Confirmed: wiki uses a session cookie named 'wikisession'. The affected users had an old cookie from a pre-migration session. Clearing cookies in the browser resolved the loop for all three users."},
        ],
        "resolution": "Root cause: stale wikisession cookies from pre-SAML migration. Workaround: clear browser cookies for wiki.corp.example.com. Permanent: IT added a notice to clear cookies after SSO migration.",
        "tags": ["saml", "sso", "login-loop", "cookie"],
        "status": "Resolved",
        "closed_date": "2025-11-07"
    },
    {
        "ticket_id": "TKT-100009",
        "title": "API token for monitoring-bot expired, Datadog alerts silent",
        "domain": "identity",
        "description": "On-call engineer noticed no Datadog alerts for 48 hours. Investigation reveals monitoring-bot's API token expired 2025-12-01 (90-day policy). All synthetic monitors now failing silently.",
        "comments": [
            {"author": "sre.team", "text": "Created new API token in Datadog UI. Updated Vault: `vault kv put secret/svc/monitoring-bot/datadog-token value=<new-token>`. Restarted monitoring-bot service to pick up new secret."},
            {"author": "sre.team", "text": "Added calendar reminder for 75-day mark to rotate proactively. Also added Datadog monitor on the bot's own heartbeat so silent failures alert on PagerDuty."},
        ],
        "resolution": "Rotated Datadog API token. Updated in Vault and restarted monitoring-bot. Added proactive rotation reminder and heartbeat monitor.",
        "tags": ["api-token", "datadog", "monitoring", "expiry"],
        "status": "Resolved",
        "closed_date": "2025-12-03"
    },

    # APP SUPPORT
    {
        "ticket_id": "TKT-100010",
        "title": "orders-service pods in CrashLoopBackOff after 14:00 deployment",
        "domain": "app_support",
        "description": "After the 14:00 deployment of orders-service v2.4.1, 3 of 5 pods entered CrashLoopBackOff. 40% of order processing requests failing with 503. Rollback being considered.",
        "comments": [
            {"author": "sre.team", "text": "`kubectl logs orders-service-abc12 --previous` shows: 'java.lang.OutOfMemoryError: Java heap space' at startup. New version loads a pre-computed recommendations cache at startup. Cache is 1.8GB, but memory limit is 1.5GB."},
            {"author": "dev.team", "text": "Confirmed: the recommendations cache loading was added in v2.4.1 without updating K8s memory limit. Options: (1) rollback, (2) increase memory limit and redeploy. Going with option 2 — updating limit to 3GB."},
        ],
        "resolution": "Updated orders-service K8s memory limit from 1.5GB to 3GB. Redeployed v2.4.1. All 5 pods running healthy. Recommendation cache loads in ~40s at startup.",
        "tags": ["kubernetes", "oom", "crashloop", "memory-limit"],
        "status": "Resolved",
        "closed_date": "2025-10-22"
    },
    {
        "ticket_id": "TKT-100011",
        "title": "Checkout API latency spike — P99 exceeding 30 seconds",
        "domain": "app_support",
        "description": "Checkout API P99 latency increased from 800ms to 32 seconds starting 09:15. P50 is 1.2s (normal). User-facing error rate at 8%. No deployment in the last 24 hours.",
        "comments": [
            {"author": "sre.team", "text": "Gateway logs show checkout-service upstream response times: P50 1.1s, P99 31s. Direct check to checkout-service bypassing gateway shows same pattern — not a gateway issue."},
            {"author": "dev.team", "text": "Found the issue: inventory-service (a downstream dependency of checkout) has a slow query caused by a full table scan. Index on orders.created_at was dropped during last night's schema migration. The P99 cases are users with large order histories."},
            {"author": "dba.team", "text": "Re-created index: `CREATE INDEX CONCURRENTLY idx_orders_created_at ON orders(created_at)`. Build took 4 minutes. P99 back to 850ms."},
        ],
        "resolution": "Missing database index on orders.created_at caused full table scans for inventory lookups. Recreated the index. P99 returned to baseline.",
        "tags": ["latency", "database", "index", "api-gateway"],
        "status": "Resolved",
        "closed_date": "2025-11-18"
    },
    {
        "ticket_id": "TKT-100012",
        "title": "Redis cluster memory at 98% — cache evictions causing DB overload",
        "domain": "app_support",
        "description": "Redis memory at 98%. Cache hit rate dropped from 94% to 61% as keys are being evicted. Database is now receiving 3x normal query load and response times have degraded. Session keys appear to be accumulating without expiry.",
        "comments": [
            {"author": "sre.team", "text": "Scanned keys: found 2.1M session keys with TTL of -1 (no expiry). These are from Spring Session integration that was deployed 6 weeks ago without configuring session timeout."},
            {"author": "dev.team", "text": "Added `spring.session.timeout=3600` to application.properties. Set Redis maxmemory-policy to allkeys-lru. Manually flushed stale session keys: `redis-cli --scan --pattern 'spring:session:sessions:*' | xargs redis-cli del`."},
        ],
        "resolution": "Configured Spring Session timeout (3600s). Set allkeys-lru eviction policy. Deleted 2.1M stale session keys. Cache memory at 41%. DB query load normalized.",
        "tags": ["redis", "memory", "session", "eviction", "spring"],
        "status": "Resolved",
        "closed_date": "2025-09-05"
    },
    {
        "ticket_id": "TKT-100013",
        "title": "TLS certificate expired on payments-gateway causing transaction failures",
        "domain": "app_support",
        "description": "Payments gateway returning 'SSL_ERROR_HANDSHAKE_FAILURE_ALERT' to mobile clients. Web clients showing certificate error. Certificate on payments-gw.corp.example.com expired at 02:00 UTC today. P1 incident declared.",
        "comments": [
            {"author": "sre.team", "text": "Confirmed: `openssl s_client -connect payments-gw.corp.example.com:443` shows 'Certificate expired' with expiry 2025-09-15 02:00:00 UTC. Requested emergency cert from IT-PKI via P1 ticket."},
            {"author": "pki.team", "text": "Emergency cert issued within 2 hours. Nginx updated: replaced cert at /etc/nginx/ssl/payments-gw.crt, ran `nginx -t && nginx -s reload`. Payments resuming."},
        ],
        "resolution": "Emergency TLS certificate issued and installed. Payments resumed. Root cause: cert was not in the rotation monitoring system. Added to cert inventory with 30-day alert.",
        "tags": ["tls", "certificate", "expired", "payments", "nginx"],
        "status": "Resolved",
        "closed_date": "2025-09-15"
    },
    {
        "ticket_id": "TKT-100014",
        "title": "Connection pool exhaustion on reporting-service database",
        "domain": "app_support",
        "description": "reporting-service throwing 'HikariPool-1 - Connection is not available, request timed out after 30000ms' errors starting 16:00. Coincides with end-of-month report generation cron job.",
        "comments": [
            {"author": "dev.team", "text": "Thread dump shows 12 threads stuck in database calls. The monthly report query runs for 8-12 minutes, holding connections open. Pool size is 10. During the batch run, all 10 connections are held by report queries, blocking all other requests."},
            {"author": "dev.team", "text": "Two-part fix: (1) added connection timeout query hint to report queries (`SET statement_timeout = '600000'`), (2) increased pool maximumPoolSize to 20, (3) separated report queries to a second pool with dedicated connections."},
        ],
        "resolution": "Increased HikariCP pool to 20. Created separate pool for batch reporting queries. Set 10-minute statement timeout on report queries to prevent runaway holds.",
        "tags": ["hikari", "connection-pool", "database", "batch"],
        "status": "Resolved",
        "closed_date": "2025-10-31"
    },
    {
        "ticket_id": "TKT-100015",
        "title": "JVM heap OOM on recommendation-engine after model cache load",
        "domain": "app_support",
        "description": "recommendation-engine service OOM-killed every 4 hours. Heap dumps show a 2.8GB retained object under ModelCacheManager. Service was recently updated to cache pre-computed embeddings for the new personalization feature.",
        "comments": [
            {"author": "ml.team", "text": "Heap dump analysis in Eclipse MAT: ModelCacheManager retains a HashMap of 900k user embedding vectors (float[512] each = ~2KB per entry). Total: 1.8GB just for embeddings, plus overhead."},
            {"author": "dev.team", "text": "Solution: (1) Moved embedding storage to Redis with TTL=86400 instead of in-heap HashMap. (2) Raised heap from -Xmx2g to -Xmx4g as interim measure while Redis migration deploys."},
        ],
        "resolution": "Moved ModelCacheManager from in-heap storage to Redis. Interim heap increase to -Xmx4g. OOM crashes stopped. Memory usage stabilized at 1.2GB heap.",
        "tags": ["jvm", "oom", "heap-dump", "cache", "recommendation"],
        "status": "Resolved",
        "closed_date": "2025-08-20"
    },
]

# ─── INCIDENTS (10) ───────────────────────────────────────────────────────────
INCIDENTS = [
    {
        "incident_id": "INC-0001",
        "title": "BGP route leak caused 45-minute partial internet outage",
        "severity": "SEV1",
        "domain": "networking",
        "summary": "A misconfigured route-map on PE02 caused BGP routes to be advertised to the wrong AS. 30% of internet-bound traffic was blackholed for 45 minutes.",
        "root_cause": "Route-map 'EXPORT_TO_UPSTREAM' was missing a deny entry for the 10.0.0.0/8 summary. During a BGP peer reset, private routes were briefly leaked to the upstream provider AS before the route-map was re-evaluated.",
        "remediation": "Removed the offending BGP session briefly using `neighbor 203.0.113.1 shutdown` to stop the leak, corrected the route-map, and re-enabled the session. Traffic restored after 8 minutes.",
        "prevention": "Added BGP prefix lists as a secondary defense alongside route-maps. Now running `show ip bgp neighbors <peer> advertised-routes` validation in change procedure checklist.",
        "date": "2025-06-15"
    },
    {
        "incident_id": "INC-0002",
        "title": "OSPF reconvergence loop caused 20-minute site isolation",
        "severity": "SEV2",
        "domain": "networking",
        "summary": "Router area reconfig during maintenance introduced an OSPF area mismatch that caused the distribution layer to oscillate between two routing states for 20 minutes.",
        "root_cause": "One router in the distribution layer was incorrectly configured as area 0 while adjacent routers expected area 1. The MTU was also mismatched at 9000 vs 1500 bytes, preventing stable adjacency.",
        "remediation": "Fixed area configuration to match. Added `ip ospf mtu-ignore` as a transitional measure. Adjacencies restabilized within 90 seconds.",
        "prevention": "Added OSPF configuration validation check to change management procedure. MTU consistency check added to pre-change checklist.",
        "date": "2025-08-03"
    },
    {
        "incident_id": "INC-0003",
        "title": "Payment service outage: database connection pool exhaustion",
        "severity": "SEV1",
        "domain": "app_support",
        "summary": "Payment service went down for 2.5 hours due to database connection pool exhaustion triggered by a slow query introduced in release 3.2.1.",
        "root_cause": "Release 3.2.1 added a new report query without using a connection timeout. During peak traffic, 15 slow instances of this query held all pool connections (pool size: 20). Subsequent requests queued until timeout.",
        "remediation": "Emergency rollback of 3.2.1. Pool size increased from 20 to 50 as immediate mitigation. Root query fixed with a covering index in v3.2.2.",
        "prevention": "Added statement_timeout=30s to all connection strings. Pool exhaustion alerts added at 80% utilization. Mandatory load test for any PR touching database queries.",
        "date": "2025-03-03"
    },
    {
        "incident_id": "INC-0004",
        "title": "Okta SSO cert expiry caused 3-hour authentication outage",
        "severity": "SEV1",
        "domain": "identity",
        "summary": "The Okta signing certificate used by 14 SAML service providers expired without warning, locking out all users from SSO-integrated applications for 3 hours.",
        "root_cause": "The Okta signing certificate had a 3-year TTL and was originally provisioned in 2022. No automated expiry monitoring existed. The certificate expired at 00:00 UTC, affecting early-shift users first.",
        "remediation": "Generated new signing certificate in Okta. Updated SP metadata (cert fingerprint) for all 14 integrations manually. Users regained access within 3 hours as integrations were fixed one by one.",
        "prevention": "Added Okta signing cert to TLS certificate inventory with 60-day alert. Created runbook for SP metadata updates. Reduced cert lifetime to 1 year on renewal.",
        "date": "2025-04-19"
    },
    {
        "incident_id": "INC-0005",
        "title": "Kubernetes memory limits caused cascading pod restarts in prod",
        "severity": "SEV2",
        "domain": "app_support",
        "summary": "12 microservices experienced CrashLoopBackOff simultaneously after a Helm chart update incorrectly set memory limits to 512Mi for all services regardless of actual requirements.",
        "root_cause": "A shared Helm chart template had a default memory limit value (`512Mi`) that was applied globally when the chart was upgraded. Services with actual requirements of 1-4GB were immediately OOMKilled.",
        "remediation": "Helm rollback to previous chart version (`helm rollback infra-services 0`) restored correct limits. Affected pods recovered within 5 minutes.",
        "prevention": "Memory limits are now service-specific values in values.yaml rather than chart defaults. Added pre-deploy validation that checks limits against known baselines.",
        "date": "2025-07-22"
    },
    {
        "incident_id": "INC-0006",
        "title": "Redis cluster split-brain caused duplicate order processing",
        "severity": "SEV2",
        "domain": "app_support",
        "summary": "During a Redis cluster node failure, a split-brain condition led to two Redis primary nodes accepting writes for the same key space, causing 340 orders to be processed twice.",
        "root_cause": "Redis sentinel quorum was set to 1 (minimum), allowing a single sentinel to promote a replica to primary even without confirmation from other sentinels. Network partition isolated one sentinel, which incorrectly promoted a new primary.",
        "remediation": "Fenced the isolated primary using Redis AUTH rotation. Reconciled duplicated orders with finance (340 items refunded). Increased sentinel quorum to 2.",
        "prevention": "Changed `sentinel quorum` to 2 (majority of 3 sentinels). Added sentinel quorum checks to infrastructure monitoring. Post-mortem review identified need for idempotency keys in order processing.",
        "date": "2025-05-11"
    },
    {
        "incident_id": "INC-0007",
        "title": "Schema migration rollback required DBA intervention",
        "severity": "SEV2",
        "domain": "app_support",
        "summary": "Release 4.1 included a schema migration that dropped a non-null column used by a legacy batch job. The application rollback succeeded but the DB migration had no rollback script, requiring manual DBA recovery.",
        "root_cause": "Migration V42 dropped column `legacy_batch_flag`. The column was referenced by a nightly batch job not covered by the application's test suite. The batch job ran at 23:00, 3 hours after deployment, and failed.",
        "remediation": "DBA restored the column from the pre-migration backup snapshot (15-minute RPO). Batch job recovered. Column retained with deprecation notice.",
        "prevention": "Mandatory down-migration (rollback SQL) for all destructive schema changes. Added legacy batch jobs to integration test suite. Column deletion now requires 2-sprint deprecation period.",
        "date": "2025-09-09"
    },
    {
        "incident_id": "INC-0008",
        "title": "Corporate VPN mass disconnection due to DPD misconfiguration",
        "severity": "SEV2",
        "domain": "networking",
        "summary": "1,200 remote workers disconnected from VPN simultaneously during a 90-minute period when WAN latency spiked during a fiber maintenance window. DPD timers were too aggressive.",
        "root_cause": "DPD timeout was set to 30 seconds with 5 retries. During the maintenance window, WAN latency reached 45-60ms. The DPD probes timed out before the latency normalized, causing the VPN gateway to declare peers dead and tear down tunnels.",
        "remediation": "Increased DPD timeout to 120s and retries to 10. This was applied via rolling config push to the VPN gateway cluster. Reconnections completed within 20 minutes.",
        "prevention": "DPD parameters now based on WAN SLA (2x the WAN SLA RTT). Added maintenance window suppression for DPD alerts. Created pre-maintenance checklist including VPN DPD review.",
        "date": "2025-10-08"
    },
    {
        "incident_id": "INC-0009",
        "title": "MFA system outage locked out 400 users for 90 minutes",
        "severity": "SEV1",
        "domain": "identity",
        "summary": "Okta MFA enforcement policy change was accidentally applied to all users including service accounts, causing 400 concurrent authentication failures and locking service accounts.",
        "root_cause": "Policy change intended for 'External Contractors' group was applied to 'All Users' due to an error in the Okta admin UI dropdown. Service accounts don't have MFA enrolled, so all service account logins began failing simultaneously.",
        "remediation": "Reverted policy to previous state via Okta admin. Unlocked 12 service accounts. Manual outreach to human users to re-establish sessions.",
        "prevention": "Okta policy changes now require a second admin to confirm the target group before saving. Staging environment Okta tenant used for all policy testing first.",
        "date": "2025-11-30"
    },
    {
        "incident_id": "INC-0010",
        "title": "JVM metaspace exhaustion caused service restarts in analytics cluster",
        "severity": "SEV3",
        "domain": "app_support",
        "summary": "analytics-service restarted 14 times over 6 hours due to Metaspace exhaustion. The service uses dynamic class loading and had no MetaspaceSize limit configured.",
        "root_cause": "The Groovy script execution engine in analytics-service generates new classes at runtime for each unique user script. With no Metaspace limit, the JVM grew indefinitely until the OS killed the process.",
        "remediation": "Set `-XX:MaxMetaspaceSize=512m` and `-XX:MetaspaceSize=256m`. Added `-XX:+CMSClassUnloadingEnabled` to allow class unloading during GC. Restarts stopped immediately.",
        "prevention": "JVM Metaspace limits added to all services using dynamic class loading. Metaspace usage metric added to APM dashboard with alert at 80% of limit.",
        "date": "2025-06-28"
    },
]

# ─── TRAINING TICKETS (50) ────────────────────────────────────────────────────
TRAIN_TICKETS = [
    # EASY — networking (cite 1 KB article or 1 past ticket)
    {"ticket_id": "TRAIN-00001", "domain": "networking", "difficulty": "easy", "is_unanswerable": False,
     "title": "BGP peer 10.0.0.1 in Active state, session not establishing",
     "description": "Router CORE-02 BGP session to peer 10.0.0.1 (AS65100) has been in Active state for 2 hours. Ping to peer IP works. No previous issues with this peer. How do I diagnose and fix?",
     "gold_cited_ids": ["KB-00001"],
     "gold_resolution": "Run `show bgp neighbors 10.0.0.1` to confirm state and check for NOTIFICATION messages. Verify AS number matches with `neighbor 10.0.0.1 remote-as 65100`. Check for ACLs blocking TCP port 179 both inbound and outbound. Fix any mismatch found, then use `clear ip bgp 10.0.0.1 soft` to re-establish."},

    {"ticket_id": "TRAIN-00002", "domain": "networking", "difficulty": "easy", "is_unanswerable": False,
     "title": "Users on floor 4 getting APIPA addresses (169.254.x.x)",
     "description": "15 users on floor 4 (VLAN 40, 10.40.0.0/24) are getting APIPA addresses this morning. Network team confirmed DHCP server is running. Yesterday floor 4 switches were patched.",
     "gold_cited_ids": ["KB-00006"],
     "gold_resolution": "Check `show ip dhcp pool` for VLAN 40 scope utilization. If near 100%, check for stale leases using `show ip dhcp binding` cross-referenced against the ARP table. Clear stale bindings with `clear ip dhcp binding <ip>`. If the pool isn't exhausted, check that DHCP helper-address is configured on the VLAN 40 SVI after last night's switch patching."},

    {"ticket_id": "TRAIN-00003", "domain": "networking", "difficulty": "easy", "is_unanswerable": False,
     "title": "Cannot resolve corp-sharepoint.corp.example.com from laptop on corporate wifi",
     "description": "My laptop on the office wifi can't reach corp-sharepoint.corp.example.com. The website loads fine on my phone using mobile data. Ping by IP works. nslookup shows 'server can't find corp-sharepoint.corp.example.com: NXDOMAIN'.",
     "gold_cited_ids": ["KB-00003"],
     "gold_resolution": "Run `nslookup corp-sharepoint.corp.example.com 10.10.1.53` directly targeting the corp DNS server. If this works, the laptop is using a wrong DNS server (likely 8.8.8.8). Check DHCP-assigned DNS: `ipconfig /all`. Fix: update DHCP option 6 for the wifi VLAN to use 10.10.1.53, then run `ipconfig /flushdns && ipconfig /renew` on the laptop."},

    {"ticket_id": "TRAIN-00004", "domain": "networking", "difficulty": "easy", "is_unanswerable": False,
     "title": "High CRC error rate on GigabitEthernet0/1 of access switch SW-FLOOR2-01",
     "description": "Network monitoring alert: 1.8% CRC error rate on GigabitEthernet0/1 of SW-FLOOR2-01 for the past 3 hours. Port connects to a workstation. The workstation reports slow network but no drops. What should I check?",
     "gold_cited_ids": ["KB-00007"],
     "gold_resolution": "First verify duplex settings: `show interface GigabitEthernet0/1 | include duplex` — if mismatch (e.g., switch is full-duplex, workstation is half or auto), this is the most likely cause. Hard-set: `speed 1000` and `duplex full` on the switch port. If duplex is OK, inspect the patch cable; replace with a Cat5e or Cat6 cable. Clear counters and monitor for 10 min after each change."},

    {"ticket_id": "TRAIN-00005", "domain": "networking", "difficulty": "easy", "is_unanswerable": False,
     "title": "F5 pool members showing red for app-backend-pool",
     "description": "app-backend-pool on the F5 LTM has all 4 members marked down. The application backend servers are running and healthy — direct curl works. The pool went down after the backend team added HTTPS redirect (HTTP to HTTPS) to their nginx config.",
     "gold_cited_ids": ["KB-00004"],
     "gold_resolution": "The F5 HTTP health monitor is failing because the backend now redirects HTTP to HTTPS (301), but the monitor expects a 200. Fix: update the pool monitor to HTTPS type with `send: HEAD /health HTTP/1.1\\r\\nHost: <backend-hostname>\\r\\n\\r\\n` and `recv: 200 OK`. Alternatively, create a dedicated health endpoint that responds 200 on HTTP without redirect."},

    {"ticket_id": "TRAIN-00006", "domain": "networking", "difficulty": "easy", "is_unanswerable": False,
     "title": "Site-to-site VPN to branch keeps dropping, reestablishes after a few minutes",
     "description": "VPN tunnel to our Singapore branch office (peer 198.51.100.22) drops every 20-40 minutes and reestablishes on its own. Users can work but experience interruptions. WAN latency to Singapore averages 180ms.",
     "gold_cited_ids": ["KB-00005"],
     "gold_resolution": "With 180ms WAN latency, the default DPD timeout of 30s with 5 retries may be too aggressive if there are occasional latency spikes. Check DPD config: increase `dpd 60 retry 10`. Also verify IKE proposal parameters match on both sides using `show crypto ikev2 sa detail` and compare algorithms and DH group with the Singapore end. Most likely cause: DPD timeout given the high-latency path."},

    {"ticket_id": "TRAIN-00007", "domain": "networking", "difficulty": "easy", "is_unanswerable": False,
     "title": "OSPF adjacency stuck in EXSTART between dist-01 and dist-02",
     "description": "Two distribution layer routers that were previously in FULL OSPF state are now stuck in EXSTART. This happened after a maintenance window where interface MTU was changed on dist-01 for jumbo frames.",
     "gold_cited_ids": ["KB-00002"],
     "gold_resolution": "EXSTART stuck state is the classic symptom of MTU mismatch. During maintenance, dist-01's interface MTU was changed but dist-02 was not updated. OSPF DBD packets use full MTU and fail to exchange. Fix: either align MTUs on both sides, or add `ip ospf mtu-ignore` on both interfaces as a workaround. After the fix, the adjacency should reach FULL state within 40 seconds (dead timer expiry)."},

    # EASY — identity (cite 1 KB article or 1 past ticket)
    {"ticket_id": "TRAIN-00008", "domain": "identity", "difficulty": "easy", "is_unanswerable": False,
     "title": "User jsmith locked out of AD, keeps getting locked again after unlock",
     "description": "User jsmith's AD account keeps getting locked every 10-15 minutes. I unlock it but it locks again. He changed his password last week. He uses Windows laptop, Outlook, Teams, and a shared network drive.",
     "gold_cited_ids": ["KB-00008"],
     "gold_resolution": "Recurring lockouts indicate a service or device is using the old password. Use Microsoft's LockoutStatus.exe or check Event ID 4740 on the PDC Emulator to find the source computer. Common culprits: mapped network drive with saved credentials, Outlook profile with cached credentials, or a mobile device. Once source is found, update the password there. The user should also run `cmdkey /list` to see all stored credentials and clear outdated ones."},

    {"ticket_id": "TRAIN-00009", "domain": "identity", "difficulty": "easy", "is_unanswerable": False,
     "title": "New employee can't log into Salesforce — account not provisioned",
     "description": "New hire Sarah Williams started today but cannot log into Salesforce. Her Okta account exists and she can log into Okta SSO. Salesforce login gives 'Invalid username, password, security token; or user locked out'.",
     "gold_cited_ids": ["KB-00009"],
     "gold_resolution": "Salesforce user was not provisioned through SCIM. Check Okta Admin > Salesforce app > Assignments to confirm Sarah is assigned. If assigned but not provisioned, check Okta System Log for SCIM push errors. Manual fix: in Okta, click 'Force Sync' for the Salesforce app for this user, which will push a SCIM POST /Users request to create her Salesforce account. Verify she appears in Salesforce Setup > Users after sync."},

    {"ticket_id": "TRAIN-00010", "domain": "identity", "difficulty": "easy", "is_unanswerable": False,
     "title": "User locked out of MFA, lost phone with authenticator app",
     "description": "User Michael Torres lost his phone. He has Google Authenticator installed for MFA. He cannot log into any corporate systems. Please reset his MFA. User has been verified in person with badge.",
     "gold_cited_ids": ["KB-00011"],
     "gold_resolution": "In Okta Admin Console, go to Directory > People, search for Michael Torres. Click the user > More Actions > Reset Multifactor. Before confirming, terminate all his active sessions under 'Current Sessions'. After reset, Michael will receive an activation email to re-enroll MFA. Log this action in the IT ticket with: user name, time, admin performing reset, and verification method (in-person badge scan)."},

    {"ticket_id": "TRAIN-00011", "domain": "identity", "difficulty": "easy", "is_unanswerable": False,
     "title": "svc-backup-prod password expired — backup jobs failing",
     "description": "Nightly backup jobs failing with authentication errors since 02:00 today. Service account svc-backup-prod password expired per 90-day policy. Account is used by the Veeam backup software and a cron job on backup-server-01.",
     "gold_cited_ids": ["KB-00012"],
     "gold_resolution": "Before changing the password, update it in CyberArk/Vault first. Then rotate in AD: `Set-ADAccountPassword -Identity svc-backup-prod -NewPassword (ConvertTo-SecureString '<new-pw>' -AsPlainText -Force) -Reset`. Update Veeam Backup & Replication: Veeam Console > Managed Servers > right-click > Properties > update credentials. Update the cron job: edit `/etc/backup/config` on backup-server-01. Verify next backup run succeeds before closing the ticket."},

    {"ticket_id": "TRAIN-00012", "domain": "identity", "difficulty": "easy", "is_unanswerable": False,
     "title": "API token for github-integration-bot returns 401 — CI pipeline failing",
     "description": "GitHub integration CI pipeline has been failing for 2 days with HTTP 401. The github-integration-bot token is used in Jenkins to push build status to GitHub. Token was working fine last month.",
     "gold_cited_ids": ["KB-00013"],
     "gold_resolution": "The GitHub PAT has likely expired (90-day policy). Generate a new token in GitHub: Settings > Developer Settings > Personal access tokens > Fine-grained tokens. Set scopes to repo status, read:packages. Update in Vault: `vault kv put secret/svc/github-integration-bot/token value=<new-token>`. Update Jenkins credential: Manage Jenkins > Credentials > update the github-api-token credential. Verify by manually triggering the CI pipeline. Revoke the old token after confirming the new one works."},

    # EASY — app_support
    {"ticket_id": "TRAIN-00013", "domain": "app_support", "difficulty": "easy", "is_unanswerable": False,
     "title": "notification-service pods CrashLoopBackOff after new deployment",
     "description": "notification-service v1.5.2 just deployed. 2 of 3 pods are in CrashLoopBackOff immediately. Deployment was not like this in staging. How do I diagnose?",
     "gold_cited_ids": ["KB-00017"],
     "gold_resolution": "Get the previous container logs with `kubectl logs <pod-name> --previous` — this shows what happened before the crash. Check `kubectl describe pod <pod-name>` for the exit reason under Events. Common causes: OOMKilled (check actual memory limit vs usage), missing ConfigMap or Secret reference (check env variable sections), or liveness probe failing too quickly (increase `initialDelaySeconds`). If the error is 'CreateContainerConfigError', a referenced Secret or ConfigMap doesn't exist in the namespace."},

    {"ticket_id": "TRAIN-00014", "domain": "app_support", "difficulty": "easy", "is_unanswerable": False,
     "title": "search-service high latency after Redis memory alert",
     "description": "search-service response times degraded from 50ms to 800ms starting an hour ago. Coincides with a Redis memory alert showing 95% usage. Redis has allkeys-lru eviction. What is happening and how to fix?",
     "gold_cited_ids": ["KB-00019"],
     "gold_resolution": "Redis is actively evicting keys under LRU policy, causing cache misses that fall through to the database. Check `redis-cli INFO stats | grep evicted_keys` for the eviction count. Check for keys without TTL: `redis-cli --scan --pattern '*' | xargs -I{} redis-cli ttl {} | grep -c '^-1'`. Quick relief: `redis-cli CONFIG SET maxmemory 4gb` if server has headroom, or identify and flush stale key patterns. Long-term: add TTL to all keys written by search-service."},

    {"ticket_id": "TRAIN-00015", "domain": "app_support", "difficulty": "easy", "is_unanswerable": False,
     "title": "analytics-service intermittently OOM crashing every few hours",
     "description": "analytics-service has been restarting with OOMKilled every 4-6 hours. Heap dumps show large arrays under ModelCacheManager. The service was recently updated to add a new ML model feature that pre-loads embeddings.",
     "gold_cited_ids": ["KB-00015"],
     "gold_resolution": "Analyze the heap dump with Eclipse MAT — run the 'Leak Suspects' report and look at the dominator tree under ModelCacheManager. If the cache is storing large in-heap objects, set `maximumSize` on any Guava/Caffeine cache, or move the cache to Redis with a TTL. As an immediate mitigation, increase heap: add `-Xmx4g -XX:+HeapDumpOnOutOfMemoryError -XX:HeapDumpPath=/var/log/heapdumps/` to JVM flags. This buys time to fix the underlying cache design."},

    {"ticket_id": "TRAIN-00016", "domain": "app_support", "difficulty": "easy", "is_unanswerable": False,
     "title": "TLS certificate expired on internal-api.corp.example.com",
     "description": "Multiple services reporting SSL handshake errors with internal-api.corp.example.com. Users getting certificate error in browser. Need to renew the certificate urgently. Certificate expired today.",
     "gold_cited_ids": ["KB-00014"],
     "gold_resolution": "Generate a CSR: `openssl req -new -newkey rsa:2048 -nodes -keyout internal-api.key -out internal-api.csr`. Submit to IT-PKI via ServiceNow under 'Certificate > Internal PKI' — for an expired cert causing outage, mark as P1 for 4-hour SLA. Once cert arrives, install: update `ssl_certificate` and `ssl_certificate_key` in nginx config, run `nginx -t` to validate, then `nginx -s reload`. Verify with `openssl s_client -connect internal-api.corp.example.com:443`."},

    # EASY — a mix
    {"ticket_id": "TRAIN-00017", "domain": "app_support", "difficulty": "easy", "is_unanswerable": False,
     "title": "connection pool timeout errors in inventory-service",
     "description": "inventory-service throwing 'Connection is not available, request timed out after 30000ms' errors under load. Database server is healthy. This happens during peak hour 18:00-19:00 daily.",
     "gold_cited_ids": ["KB-00016"],
     "gold_resolution": "Check Prometheus metric `hikaricp_connections_active` vs `hikaricp_connections_max` during peak. If saturated, take a thread dump (`kill -3 <pid>`) and look for database-blocking threads. Confirm pool config: `maximumPoolSize` should be at least (peak concurrent requests × avg query duration). Immediate fix: increase `maximumPoolSize`. Add `connectionTimeout=30000`, `idleTimeout=600000`, `leakDetectionThreshold=60000` to HikariCP config. Enable `hikaricp.connections.active` alerting at 80% of max pool size."},

    {"ticket_id": "TRAIN-00018", "domain": "app_support", "difficulty": "easy", "is_unanswerable": False,
     "title": "prod deployment of billing-service failed, need to rollback",
     "description": "billing-service v2.8.0 just deployed and error rate jumped to 15%. We need to roll back to the previous version immediately. Service is deployed via Helm.",
     "gold_cited_ids": ["KB-00020"],
     "gold_resolution": "Execute `helm rollback billing-service 0` to roll back to the previous Helm revision (0 means previous). Watch progress: `kubectl rollout status deployment/billing-service --timeout=120s`. Verify in `kubectl get pods` that all pods are Running with recent age. Post to #incidents Slack channel: what happened, rollback executed, impact window. Check `helm history billing-service` to confirm revision before filing post-mortem."},

    {"ticket_id": "TRAIN-00019", "domain": "identity", "difficulty": "easy", "is_unanswerable": False,
     "title": "SAML SSO broken for jira.corp.example.com — redirect loop",
     "description": "Several users reporting redirect loop when logging into jira.corp.example.com using SSO. They get sent to Okta, authenticate, sent back to Jira, and then immediately redirected back to Okta. Direct username/password login works.",
     "gold_cited_ids": ["KB-00010"],
     "gold_resolution": "Install SAML Tracer in Chrome and record the login attempt. Look for the SAMLResponse POST to Jira's ACS URL. Check the `<Conditions NotBefore NotOnOrAfter>` timestamps — if expired, it's clock skew. Also decode the SAMLResponse and check the `InResponseTo` attribute; if Jira's session is generating duplicate AuthnRequest IDs, clearing browser cookies for jira.corp.example.com resolves the loop. Also verify Jira's Entity ID in Okta config exactly matches Jira's configured Audience (case-sensitive)."},

    {"ticket_id": "TRAIN-00020", "domain": "networking", "difficulty": "easy", "is_unanswerable": False,
     "title": "OSPF route to 10.20.0.0/16 missing after router replacement",
     "description": "Router RTR-BRANCH-05 was replaced with new hardware over the weekend. After replacement, OSPF routes to 10.20.0.0/16 are missing from the routing table. The new router is in the same physical location. Can reach adjacent routers via ICMP.",
     "gold_cited_ids": ["KB-00002"],
     "gold_resolution": "Run `show ip ospf neighbor` — if the new router shows no neighbors or neighbors in INIT state, check area configuration first. Common issue after hardware replacement: the new router was configured with the wrong OSPF area ID. Also check interface MTU — if the new hardware has different default MTU, EXSTART can get stuck. Use `debug ip ospf adj` to see the exact adjacency failure reason. If area and MTU match, check authentication MD5 keys."},

    # MEDIUM — networking (cite 2+ items)
    {"ticket_id": "TRAIN-00021", "domain": "networking", "difficulty": "medium", "is_unanswerable": False,
     "title": "OSPF adjacency failing AND external BGP route not appearing",
     "description": "After adding a new router to the network, two issues: (1) OSPF adjacency with the new router is stuck in EXSTART, (2) The BGP route to 10.99.0.0/24 that OSPF should redistribute is missing. Both issues need root cause.",
     "gold_cited_ids": ["KB-00001", "KB-00002"],
     "gold_resolution": "Two separate root causes. For OSPF EXSTART: check MTU mismatch — the new router likely has different MTU. Fix with `ip ospf mtu-ignore` on both sides or align MTUs. Once OSPF is FULL, check if 10.99.0.0/24 appears in OSPF LSAs: `show ip ospf database external`. For the BGP issue: verify the route exists in BGP with `show bgp neighbors <peer-ip>` and that redistribution is configured: `redistribute bgp 65001 subnets` in the OSPF config. Both issues together suggest the new router config was applied from a template missing these settings."},

    {"ticket_id": "TRAIN-00022", "domain": "networking", "difficulty": "medium", "is_unanswerable": False,
     "title": "VPN tunnel flapping AND DHCP pool nearly exhausted at branch",
     "description": "Branch office Bangkok reporting two issues today: (1) VPN tunnel to HQ keeps resetting every 15 minutes, and (2) DHCP pool for the branch is 94% utilized with only 12 IPs left for 80 devices. Both issues are impacting business.",
     "gold_cited_ids": ["KB-00005", "KB-00006"],
     "gold_resolution": "Two independent issues. For VPN: compare IKE proposals between HQ and Bangkok — check `debug crypto ikev2` for NO_PROPOSAL_CHOSEN or DPD timeout messages. DH group mismatch or aggressive DPD given the network distance are most common. For DHCP: immediately `show ip dhcp binding` and cross-reference against ARP to find stale leases. Clear stale entries with `clear ip dhcp binding <ip>`. For capacity: reduce lease time from 8d to 1d as a stopgap; plan scope expansion to /23 in the next change window."},

    {"ticket_id": "TRAIN-00023", "domain": "networking", "difficulty": "medium", "is_unanswerable": False,
     "title": "BGP session flapping since firewall ACL change — similar to Bangkok VPN issue last month",
     "description": "BGP session to upstream ISP keeps going Active/Established every 2 hours, similar pattern to the Bangkok branch issue we had. Firewall team made ACL changes 3 days ago. This is the same symptom as ticket TKT-100001.",
     "gold_cited_ids": ["KB-00001", "TKT-100001"],
     "gold_resolution": "Based on TKT-100001, the most likely cause is a firewall ACL killing established TCP sessions after a timeout (the Bangkok case had 7200s = 2 hours, which matches the observed pattern). Check the recent firewall ACL change for any session timeout rules on TCP/179. The fix from TKT-100001 was adding a stateful permit rule for established TCP/179 sessions without a timeout. Run `show bgp neighbors <peer> | include BGP state|Hold time` and compare hold timer (default 180s) against the observed drop frequency."},

    {"ticket_id": "TRAIN-00024", "domain": "identity", "difficulty": "medium", "is_unanswerable": False,
     "title": "Salesforce users not deprovisioned after offboarding AND SAML cert expired",
     "description": "Security audit found two identity issues: (1) 8 offboarded employees still have active Salesforce accounts (similar to our previous Okta SCIM incident), (2) The SAML signing cert for Salesforce in Okta expires in 5 days.",
     "gold_cited_ids": ["KB-00009", "KB-00010"],
     "gold_resolution": "Two issues. For SCIM deprovisioning: check Okta System Log for deprovisioning push failures. Likely cause: 'Deactivate Users' toggle is OFF in Okta > Salesforce > Provisioning (this was the same root cause as TKT-100007). Enable it and manually trigger deprovisioning for the 8 users. For the expiring cert: in Okta, navigate to the Salesforce SAML app > Sign On > View SAML Setup Instructions. Generate a new signing certificate, download it, then update Salesforce: Setup > Single Sign-On Settings > update the Identity Provider Certificate. Update before expiry to avoid an outage."},

    {"ticket_id": "TRAIN-00025", "domain": "app_support", "difficulty": "medium", "is_unanswerable": False,
     "title": "orders-service CrashLoopBackOff AND database connection errors on related services",
     "description": "After today's deployment, orders-service pods are in CrashLoopBackOff (3 of 5 pods). Additionally, fulfillment-service (which shares the orders DB) is reporting connection pool timeouts. Both issues started at 14:00.",
     "gold_cited_ids": ["KB-00017", "KB-00016"],
     "gold_resolution": "Two related issues. For orders-service CrashLoopBackOff: `kubectl logs <pod> --previous` to see crash reason. Check Events in `kubectl describe pod` — likely OOMKilled or config error. For the connection pool issue: the CrashLoopBackOff causes orders-service to repeatedly connect and disconnect, leaving connections in the pool that aren't properly closed. Check Prometheus `hikaricp_connections_active` for fulfillment-service — if near max, the pool is being starved by zombie connections from crashing orders-service pods. Resolve the crash first, then restart fulfillment-service to flush the connection pool."},

    {"ticket_id": "TRAIN-00026", "domain": "app_support", "difficulty": "medium", "is_unanswerable": False,
     "title": "Redis memory pressure causing evictions AND API gateway 504 timeouts",
     "description": "Two linked issues: redis cluster is at 92% memory with active evictions, and the checkout API is returning 504 timeouts. The checkout service relies heavily on Redis for session and cart caching. Started 2 hours ago.",
     "gold_cited_ids": ["KB-00019", "KB-00018"],
     "gold_resolution": "Redis evictions are causing cache misses that increase checkout-service database load, which causes slow queries and timeouts at the API gateway. For Redis: immediately check `redis-cli INFO stats | grep evicted_keys` and find keys without TTL. Set `maxmemory-policy allkeys-lru` if not set. For 504s: in gateway logs check which upstream is timing out. Do not increase gateway timeout — it masks the root cause. Fix Redis memory pressure first, then verify checkout response times normalize. Long-term: add TTL to all cart/session keys in checkout-service code."},

    {"ticket_id": "TRAIN-00027", "domain": "identity", "difficulty": "medium", "is_unanswerable": False,
     "title": "service account svc-k8s-deploy locked after rotation — deployment pipeline broken",
     "description": "Deployment pipeline failed at 11:00. Symptoms: Jenkins says 'LDAP authentication failed for svc-k8s-deploy'. The account was in a password rotation window this morning. We also confirmed the Kubernetes imagePullSecret was updated but now K8s pods won't start.",
     "gold_cited_ids": ["KB-00012", "TKT-100006"],
     "gold_resolution": "Based on TKT-100006, incomplete password rotation is the root cause pattern. Unlock the account in AD: `Unlock-ADAccount -Identity svc-k8s-deploy`. Check all dependencies: (1) Jenkins — update credentials in Manage Jenkins > Credentials, (2) Kubernetes — verify imagePullSecret: `kubectl get secret regcred -n prod -o yaml | base64 -d` to confirm new password is reflected, (3) Any Ansible vault or Helm secrets referencing this account. Follow the rotation procedure in KB-00012: update Vault first, then AD, then each dependent service in sequence."},

    {"ticket_id": "TRAIN-00028", "domain": "app_support", "difficulty": "medium", "is_unanswerable": False,
     "title": "JVM OOM crashes AND high DB connection count — both on recommendation-engine",
     "description": "recommendation-engine is being OOM-killed every few hours. Additionally, DBA team reports the service holds 45 of 50 available DB connections even when idle. Heap dumps show a large object under ModelCacheManager.",
     "gold_cited_ids": ["KB-00015", "KB-00016"],
     "gold_resolution": "Two issues but same root: the ModelCacheManager is both causing heap exhaustion and holding DB connections to populate the cache on demand. From the heap dump, identify what ModelCacheManager is storing in memory (likely large float arrays for embeddings — see TKT-100015 for precedent). Fix: offload ModelCacheManager from heap to Redis with TTL. For the connection leak: add `leakDetectionThreshold=60000` to HikariCP config to identify which code path holds connections open. Increase heap to `-Xmx4g` as interim while the cache redesign deploys."},

    {"ticket_id": "TRAIN-00029", "domain": "networking", "difficulty": "medium", "is_unanswerable": False,
     "title": "DHCP pool exhausted on IoT VLAN AND high interface errors on connected switch",
     "description": "IoT VLAN 60 DHCP pool is full (200/200 leases assigned). Simultaneously, the access switch serving this VLAN shows 3.1% CRC error rate on its uplink. Both started after adding a new batch of sensors yesterday.",
     "gold_cited_ids": ["KB-00006", "KB-00007"],
     "gold_resolution": "Two issues from yesterday's addition. For DHCP: identify stale leases from retired sensors using `show ip dhcp binding` vs ARP table. Clear orphaned entries. Reduce lease time to 1 day and consider expanding scope to /23 to accommodate growth. For CRC errors: the new sensor installation may have involved cable work near the uplink — inspect the patch cable for the uplink port, check connector seating and duplex settings. Run `show interface | include CRC` baseline, fix duplex if mismatched, replace cable if errors persist after duplex fix."},

    {"ticket_id": "TRAIN-00030", "domain": "identity", "difficulty": "medium", "is_unanswerable": False,
     "title": "MFA reset needed AND AD account locked for same user — urgent, user is executive",
     "description": "Executive user David Chen is locked out of all systems. His AD account is locked (bad password count 12), AND he lost his phone with MFA app. He is verified via video call. Need both resolved within 15 minutes.",
     "gold_cited_ids": ["KB-00008", "KB-00011"],
     "gold_resolution": "Parallel actions. For AD lockout: `Unlock-ADAccount -Identity dchen` in PowerShell. Check Event ID 4740 on PDC Emulator for lockout source — likely his old phone or iPad with cached corporate credentials. For MFA: Okta Admin > Directory > People > dchen > More Actions > Reset Multifactor. First terminate all active sessions, then reset MFA. Log both actions with video call verification. Advise David to update credentials on all devices after regaining access to prevent immediate re-lockout."},

    {"ticket_id": "TRAIN-00031", "domain": "app_support", "difficulty": "medium", "is_unanswerable": False,
     "title": "billing-service API 504 timeouts AND recent deployment rollback needed",
     "description": "billing-service v3.1.0 was deployed 30 minutes ago. API gateway is now showing 504 timeouts on /billing/invoice. The previous version v3.0.8 did not have this issue. We need to investigate and potentially rollback.",
     "gold_cited_ids": ["KB-00018", "KB-00020"],
     "gold_resolution": "Collect evidence first: check API gateway logs for `upstream_response_time` on /billing/invoice requests. If upstream times out consistently (>30s), rollback is the right call. Execute: `helm rollback billing-service 0`. Monitor: `kubectl rollout status deployment/billing-service --timeout=120s`. If the 504 is intermittent (partial rollout issue), try `kubectl rollout restart deployment/billing-service` first. For root cause investigation post-rollback: compare v3.0.8 vs v3.1.0 changelogs for any database queries or external calls added to the invoice endpoint."},

    {"ticket_id": "TRAIN-00032", "domain": "app_support", "difficulty": "medium", "is_unanswerable": False,
     "title": "checkout-service high memory AND Redis cache not working after deployment",
     "description": "After deploying checkout-service v2.2, heap usage climbed from 800MB to 3.5GB in 2 hours. Additionally, Redis cache hit rate dropped from 88% to 12%, suggesting caching is broken. Both started with v2.2.",
     "gold_cited_ids": ["KB-00015", "KB-00019"],
     "gold_resolution": "Likely related: v2.2 may have introduced unbounded in-memory cache that bypasses Redis or broke the Redis TTL logic. Capture a heap dump: `jmap -dump:format=b,file=/tmp/heap.hprof <pid>`. Analyze with Eclipse MAT — if a cache data structure is the largest retained object, it was moved from Redis to heap in v2.2. Check Redis: `redis-cli INFO stats | grep hit_rate` to confirm misses. Review the v2.2 diff for cache manager changes. Quick fix: rollback or increase heap to `-Xmx4g`. Fix: ensure all cache writes use Redis with TTL — check the CacheConfig class for `maximumSize` and `ttl` settings."},

    # HARD — multi-doc, with distractors
    {"ticket_id": "TRAIN-00033", "domain": "networking", "difficulty": "hard", "is_unanswerable": False,
     "title": "Multiple network failures after datacenter fiber maintenance: BGP, OSPF, DHCP all affected",
     "description": "After last night's datacenter fiber maintenance window, three issues appeared: (1) BGP session to ISP dropping, (2) OSPF adjacency between core and dist not forming, (3) One VLAN DHCP pool showing exhaustion alerts. Maintenance involved re-patching several cables. Need systematic root cause for all three.",
     "gold_cited_ids": ["KB-00001", "KB-00002", "KB-00006"],
     "gold_resolution": "Three issues, likely caused by the cable re-patching. BGP: check for ACL rules blocking TCP 179 if cables were re-patched through a firewall segment; also check MD5 password on the BGP session. OSPF: fiber re-patching may have changed the MTU — physical cabling changes can affect VLAN configuration. Run `show ip ospf neighbor` to see stuck state. If EXSTART, check MTU and area config. DHCP: the re-patching may have disconnected some devices temporarily, causing them to request new leases without releasing old ones. Cross-reference `show ip dhcp binding` against ARP. Clear stale bindings from devices that were temporarily offline."},

    {"ticket_id": "TRAIN-00034", "domain": "identity", "difficulty": "hard", "is_unanswerable": False,
     "title": "SSO broken for multiple apps, SCIM not working, AND an API token expiry — all identity issues",
     "description": "Major identity incident: (1) Users can't SSO into Confluence and JIRA — redirect loop in both, (2) Okta SCIM provisioning stopped pushing to GitHub after midnight, (3) The monitoring API token for the identity health dashboard shows 401. All three started around midnight.",
     "gold_cited_ids": ["KB-00010", "KB-00009", "KB-00013"],
     "gold_resolution": "All three are related to an Okta signing certificate rotation that happened at midnight. For SSO: the Confluence and JIRA SAML SPs have outdated signing certificates — update the SP trust with the new Okta cert fingerprint (Settings > Security > SSO in each app). For SCIM: Okta's SCIM Bearer tokens for GitHub may have been regenerated as part of the rotation — check Okta System Log for 401 errors in GitHub SCIM push events, regenerate the token in GitHub and update in Okta. For the monitoring token: the identity health dashboard uses a service account token that was invalidated — rotate per KB-00013 zero-downtime pattern."},

    {"ticket_id": "TRAIN-00035", "domain": "app_support", "difficulty": "hard", "is_unanswerable": False,
     "title": "Platform-wide degradation: 504s, pod crashes, and Redis OOM all occurring simultaneously",
     "description": "Widespread platform issue. api-gateway showing 15% 504 error rate. Six pods in CrashLoopBackOff across three services. Redis at 99% memory with active evictions. All three issues started within 5 minutes of each other at 16:00. No deployments today.",
     "gold_cited_ids": ["KB-00017", "KB-00018", "KB-00019"],
     "gold_resolution": "This is a cascading failure starting from Redis. Redis at 99% memory caused mass evictions (KB-00019), which caused cache misses across all services, which caused downstream DB query spikes. Services with OOM-prone configurations (KB-00017) crashed under the DB load, entering CrashLoopBackOff. The pod crashes further increased Redis connection pressure from reconnection storms. Resolution priority: (1) Immediately scale Redis memory: `redis-cli CONFIG SET maxmemory 8gb`. (2) Identify and flush keys without TTL. (3) For crashed pods, check `kubectl logs --previous` — likely OOMKilled from increased memory pressure. (4) For 504s, the upstream services will recover as Redis stabilizes. Do not increase gateway timeout (KB-00018)."},

    {"ticket_id": "TRAIN-00036", "domain": "identity", "difficulty": "hard", "is_unanswerable": False,
     "title": "Service account breach suspected — need to rotate credentials and audit all usages",
     "description": "Security team flagged unusual API calls from svc-reporting-api between 03:00-04:00. Token activity from an unexpected IP. We need to: (1) immediately revoke the current token, (2) audit all services using this account, (3) rotate the AD password, (4) check if MFA was bypassed.",
     "gold_cited_ids": ["KB-00013", "KB-00012", "KB-00011"],
     "gold_resolution": "Emergency sequence: (1) Revoke the API token immediately per KB-00013 emergency revocation path — revoke first, accept brief outage, then update consumers. `DELETE /api/v1/tokens/<token-id>`. (2) Rotate AD password per KB-00012 — update Vault, then rotate in AD: `Set-ADAccountPassword -Identity svc-reporting-api -Reset`. Update all consumers: Jenkins, K8s secrets, any config files. (3) If svc-reporting-api has MFA capability check per KB-00011 — for service accounts, MFA is typically disabled; verify the account type. (4) Preserve audit logs before any cleanup. Involve the security team for forensic review. File a P1 security incident."},

    {"ticket_id": "TRAIN-00037", "domain": "networking", "difficulty": "hard", "is_unanswerable": False,
     "title": "New branch office cannot connect to corporate — VPN, BGP redistribution, and DNS all broken",
     "description": "New branch office in Hyderabad brought online today. Three issues discovered: (1) VPN tunnel shows IKE_SA_INIT OK but CHILD_SA fails, (2) BGP routes from the branch are not appearing in the core routing table, (3) Branch users cannot resolve internal corporate DNS names.",
     "gold_cited_ids": ["KB-00005", "KB-00001", "KB-00003"],
     "gold_resolution": "Three config issues for the new site. VPN CHILD_SA failure: this is Phase 2 (IPSec) failure after Phase 1 succeeded. Check traffic selectors — the branch proxy-id may not match HQ's. Verify `show crypto ipsec sa` for mismatched SA selectors. Also verify PFS group matches on both sides. BGP routing: once VPN is up, check if BGP session forms but routes aren't advertised — verify `network` statements or `redistribute connected` in BGP config. Confirm no route-map filtering the new 10.x.x.x/24 prefix. DNS: branch DHCP must point to 10.10.1.53 and 10.10.2.53 (corporate DNS), not local ISP resolvers. Check DHCP option 6 configuration on the branch DHCP server or router."},

    {"ticket_id": "TRAIN-00038", "domain": "app_support", "difficulty": "hard", "is_unanswerable": False,
     "title": "Post-deployment: OOM crashes, DB pool exhaustion, AND a needed rollback with DB migration concern",
     "description": "Release v5.0.0 was deployed 2 hours ago. Three issues: (1) user-profile service OOM-crashing, (2) orders-service DB connections exhausted, (3) We want to rollback but v5.0.0 included a DB schema migration — need to understand if rollback is safe.",
     "gold_cited_ids": ["KB-00015", "KB-00016", "KB-00020"],
     "gold_resolution": "Address in order. DB migration safety first: check `/db/migrations/` for a `V<n>__down.sql`. If exists, rollback is safe. If not, escalate to DBA — rolling back the app without rolling back the schema may break data integrity. For OOM: take heap dump before rollback to preserve evidence: `jmap -dump:format=b,file=/tmp/heap.hprof <pid>`. For connection pool: thread dump to identify holders. If DBA approves rollback: `helm rollback user-profile 0` and `helm rollback orders-service 0`. Monitor post-rollback. If no down migration: consult KB-00020 — VM-based apps use symlink rollback. For Kubernetes: rollback app, leave DB schema, add compatibility code, and schedule a proper schema rollback in next maintenance window."},

    # UNANSWERABLE (8)
    {"ticket_id": "TRAIN-00039", "domain": "networking", "difficulty": "medium", "is_unanswerable": True,
     "title": "Cisco Catalyst 9300 firmware upgrade causing LACP bundle failures",
     "description": "After upgrading Catalyst 9300 switches to IOS-XE 17.12.3, LACP bundles between the 9300 and our Nexus 7K are randomly dropping one member port every few hours. No runbook exists for 9300/Nexus cross-vendor LACP issues.",
     "gold_cited_ids": [],
     "gold_resolution": "Escalate to L2 network team and open a TAC case with Cisco. No current runbook covers IOS-XE 17.12.3 compatibility with Nexus LACP. Collect: `show lacp neighbor detail`, `show etherchannel summary`, and syslog from both platforms during a bundle drop event."},

    {"ticket_id": "TRAIN-00040", "domain": "identity", "difficulty": "medium", "is_unanswerable": True,
     "title": "FIDO2/WebAuthn passkey not working on new Lenovo T16 hardware",
     "description": "New batch of Lenovo T16 laptops cannot use FIDO2 passkeys for corporate SSO. The built-in fingerprint reader is not being recognized as a FIDO2 authenticator. Other laptop models work fine.",
     "gold_cited_ids": [],
     "gold_resolution": "Escalate to endpoint team and the identity platform team. FIDO2/WebAuthn hardware compatibility for the Lenovo T16 fingerprint reader is not covered in any current KB article. Open a ticket with the hardware vendor and check Windows Hello for Business compatibility matrix for this model."},

    {"ticket_id": "TRAIN-00041", "domain": "app_support", "difficulty": "easy", "is_unanswerable": True,
     "title": "Kafka consumer group lag spike on recommendation topic — no runbook",
     "description": "Kafka consumer group recommendation-consumer is showing 4.2M message lag on the recommendations topic, growing. The consumer is a custom Rust application. No prior incidents with this system. No runbook exists.",
     "gold_cited_ids": [],
     "gold_resolution": "Escalate to the data engineering team who owns the Rust consumer. There is no runbook for Kafka consumer lag in this system. Collect: consumer group lag via `kafka-consumer-groups.sh --describe`, consumer logs, and Kafka broker metrics for the partition assignment."},

    {"ticket_id": "TRAIN-00042", "domain": "networking", "difficulty": "easy", "is_unanswerable": True,
     "title": "Aruba ClearPass NAC blocking new IoT device category",
     "description": "Newly approved wearable sensors from vendor Bosch are being quarantined by Aruba ClearPass NAC. The device fingerprint is not in ClearPass and there's no policy for this device class. No runbook available.",
     "gold_cited_ids": [],
     "gold_resolution": "Escalate to the network access control team. No runbook or KB article covers Aruba ClearPass NAC device fingerprinting and policy configuration. The NAC team will need to add a device profile for the Bosch wearables and define an appropriate VLAN assignment policy."},

    {"ticket_id": "TRAIN-00043", "domain": "identity", "difficulty": "medium", "is_unanswerable": True,
     "title": "Entra ID Conditional Access policy blocking external contractors on new EU GDPR laptop policy",
     "description": "EU-based contractors are being blocked by a new Conditional Access policy that requires a compliant device, but contractors use personal laptops that cannot be enrolled in Intune. No runbook for this scenario.",
     "gold_cited_ids": [],
     "gold_resolution": "Escalate to the Identity Governance team. This scenario — compliant device CA policy conflicting with contractor BYO laptop policy — requires a policy exception or a separate CA policy for contractor persona. No current KB article or procedure covers this case."},

    {"ticket_id": "TRAIN-00044", "domain": "app_support", "difficulty": "medium", "is_unanswerable": True,
     "title": "GPU-accelerated ML inference service running out of VRAM — no precedent",
     "description": "New GPU inference service for the image classification feature is hitting CUDA out of memory errors (CUDA_ERROR_OUT_OF_MEMORY) on the A10G GPU. This is a new service type and there is no existing runbook or past ticket for GPU memory management.",
     "gold_cited_ids": [],
     "gold_resolution": "Escalate to the ML platform team. GPU VRAM management and CUDA OOM troubleshooting is not covered in any current KB article (existing articles cover JVM heap and Redis memory, not GPU memory). The ML platform team will need to profile GPU memory usage and configure model batching or precision reduction."},

    {"ticket_id": "TRAIN-00045", "domain": "networking", "difficulty": "easy", "is_unanswerable": True,
     "title": "SD-WAN policy change needed for new video conferencing traffic class",
     "description": "Cisco Viptela SD-WAN needs a new traffic class and QoS policy for the recently approved Zoom Rooms appliances. No existing runbook covers Viptela SD-WAN QoS policy management.",
     "gold_cited_ids": [],
     "gold_resolution": "Escalate to the SD-WAN team. No KB article or runbook covers Cisco Viptela SD-WAN traffic policy configuration. The SD-WAN team manages policy templates centrally through vManage and will need to create a new data policy for the Zoom Rooms traffic class."},

    {"ticket_id": "TRAIN-00046", "domain": "identity", "difficulty": "easy", "is_unanswerable": True,
     "title": "BeyondCorp zero-trust access request for a new OT network segment",
     "description": "Manufacturing OT network segment needs to be onboarded to the BeyondCorp zero-trust access framework. OT devices use legacy protocols (Modbus, DNP3) that are not compatible with the existing BeyondCorp agent. No runbook exists.",
     "gold_cited_ids": [],
     "gold_resolution": "Escalate to the Network Security Architecture team. OT device onboarding to BeyondCorp is not covered in any current KB article — it requires a specialized architecture review due to the legacy protocol constraints. The team will evaluate options including protocol-aware proxies or separate access control mechanisms."},

    # Fill remaining training tickets to reach 50
    {"ticket_id": "TRAIN-00047", "domain": "app_support", "difficulty": "medium", "is_unanswerable": False,
     "title": "TLS cert expired on internal microservice causing auth failures",
     "description": "Multiple services failing auth with internal-auth-service.corp.example.com. Error: 'SSL certificate expired'. The cert on this internal service expired overnight. P1 impact: all OAuth token validation is down.",
     "gold_cited_ids": ["KB-00014", "TKT-100013"],
     "gold_resolution": "This is a P1 cert expiry per the pattern in TKT-100013. Request emergency certificate from IT-PKI via ServiceNow P1 ticket (4-hour SLA for cert causing outage). While waiting: generate CSR using `openssl req -new -newkey rsa:2048 -nodes -keyout auth-svc.key -out auth-svc.csr`. Include all SANs the service uses. Once cert arrives, install and reload nginx: `nginx -t && nginx -s reload`. Add cert to the certificate inventory monitoring system immediately after resolution to prevent recurrence."},

    {"ticket_id": "TRAIN-00048", "domain": "identity", "difficulty": "medium", "is_unanswerable": False,
     "title": "Mass Okta MFA outage after policy rollout — same root cause as past incident",
     "description": "At 09:00, 200 users started reporting MFA failures. Okta system log shows policy change last night. Service accounts are also failing. This has a similar pattern to INC-0009.",
     "gold_cited_ids": ["KB-00011", "INC-0009"],
     "gold_resolution": "Based on INC-0009, a policy change was applied to too broad a group. In Okta Admin, check the MFA enforcement policy change from last night — verify target group. If it was applied to 'All Users' instead of a specific group, revert it immediately. For locked-out service accounts: per INC-0009 remediation, run through the list of service accounts and unlock each one in AD. For human users: once the policy is reverted, their next login should prompt MFA enrollment if they weren't previously enrolled. Document the fix and add the 'second admin confirmation' control from INC-0009 prevention section."},

    {"ticket_id": "TRAIN-00049", "domain": "networking", "difficulty": "medium", "is_unanswerable": False,
     "title": "All internet-facing services offline — BGP route leak suspected",
     "description": "Internet-facing services unreachable externally since 14:00. Internal-to-internet traffic fine. Suspected BGP route leak to upstream provider based on traceroute results showing traffic entering our AS but not returning. Similar to INC-0001.",
     "gold_cited_ids": ["KB-00001", "INC-0001"],
     "gold_resolution": "Per INC-0001 precedent: immediately `neighbor <peer-ip> shutdown` on the BGP session to the upstream to stop the leak. Investigate route-map configuration: `show route-map EXPORT_TO_UPSTREAM` — check for missing deny entries that are allowing private prefixes to leak. Add a deny entry for 10.0.0.0/8 and any other RFC-1918 prefixes if missing. After fixing the route-map, re-enable the session: `no neighbor <peer-ip> shutdown`. Run `show ip bgp neighbors <peer-ip> advertised-routes | grep 10\\.` to confirm no private routes are being advertised."},

    {"ticket_id": "TRAIN-00050", "domain": "app_support", "difficulty": "medium", "is_unanswerable": False,
     "title": "Deployment rollback needed but has DB schema migration — same concern as recent incident",
     "description": "We deployed catalog-service v3.5.0 which included DB migration V38 that added two tables. Service is unstable and we need to rollback. Concerned about the DB migration per INC-0007.",
     "gold_cited_ids": ["KB-00020", "INC-0007"],
     "gold_resolution": "Per INC-0007 precedent, check if V38 has a corresponding down migration script in `/db/migrations/V38__down.sql`. If yes, the rollback is safe: `helm rollback catalog-service 0`, then run the down migration through the DBA team. If no down migration exists (like INC-0007), do NOT rollback blindly — the new tables may already have data. Options: (1) Keep the app at the new version and hotfix, (2) DBA evaluates whether the new tables are empty and can be dropped safely, (3) Accept the partial rollback knowing the tables will be orphaned. Create post-mortem requirement: all destructive or additive schema changes require down migrations."},
]

# ─── EVAL TICKETS (20, ~15 mix + 5 unanswerable) ─────────────────────────────
EVAL_TICKETS = [
    {"ticket_id": "EVAL-00001", "domain": "networking", "difficulty": "easy", "is_unanswerable": False,
     "title": "OSPF neighbors flapping on core switch after hardware replacement",
     "description": "Core switch SW-CORE-03 was replaced with a new unit last night. OSPF neighbors are cycling every few minutes. Old unit had MTU 9000 (jumbo frames). New unit has default MTU.",
     "gold_cited_ids": ["KB-00002"],
     "gold_resolution": "MTU mismatch is the cause — old unit at 9000, new at 1500. Apply `ip ospf mtu-ignore` on all interconnected interfaces on the new switch, or set MTU to 9000: `system mtu jumbo 9000` on the new Cisco switch hardware. Adjacencies should restabilize within 40 seconds."},

    {"ticket_id": "EVAL-00002", "domain": "identity", "difficulty": "easy", "is_unanswerable": False,
     "title": "New joiner can't log into GitLab — account not created",
     "description": "New engineer Priya Nair started today. Can access Okta dashboard, but GitLab shows 'Invalid login or password'. IT confirmed she is assigned to the GitLab Okta app.",
     "gold_cited_ids": ["KB-00009"],
     "gold_resolution": "SCIM provisioning likely hasn't pushed her account yet. In Okta: Applications > GitLab > Assignments — find Priya Nair and click 'Force Sync'. This triggers a SCIM POST /Users to create her account. Check Okta System Log for the push result. If the push shows a 409 (user already exists with different email), investigate the GitLab user list for a conflicting account."},

    {"ticket_id": "EVAL-00003", "domain": "app_support", "difficulty": "easy", "is_unanswerable": False,
     "title": "K8s pod for data-processor failing to start — missing config",
     "description": "data-processor pod entered CrashLoopBackOff immediately after deployment. Logs are empty — pod crashes in under a second. This has never happened with this service before.",
     "gold_cited_ids": ["KB-00017"],
     "gold_resolution": "Use `kubectl logs <pod> --previous` for prior-run logs. If empty, crash is sub-second — check `kubectl describe pod <pod>` Events section for 'CreateContainerConfigError' which indicates a missing Secret or ConfigMap. Run `kubectl get secret data-processor-config -n <namespace>` — if not found, the secret wasn't created in this namespace. Create it or contact the deploying team to pre-create the referenced secrets before the pod can start."},

    {"ticket_id": "EVAL-00004", "domain": "networking", "difficulty": "medium", "is_unanswerable": False,
     "title": "VPN flapping AND BGP not establishing to same remote site",
     "description": "Remote site Pune: VPN tunnel flapping (IKE phase 1 succeeds, phase 2 fails) AND BGP session never establishes over the VPN. IT team at Pune says their equipment was just patched.",
     "gold_cited_ids": ["KB-00005", "KB-00001"],
     "gold_resolution": "Phase 2 failure (CHILD_SA): check traffic selectors in `show crypto ipsec sa` — Pune patch may have changed proxy-IDs. Verify PFS group matches. For BGP over VPN: BGP cannot establish until VPN is up and routing is correct. Once VPN phase 2 is fixed, check if BGP TCP/179 can traverse the tunnel: `telnet <peer-ip> 179 /source-ip <local-ip>`. Verify BGP timers and AS match. BGP over unstable VPN will keep getting reset — fix VPN first."},

    {"ticket_id": "EVAL-00005", "domain": "app_support", "difficulty": "medium", "is_unanswerable": False,
     "title": "payment-service DB connections exhausted AND needs emergency rollback",
     "description": "payment-service v4.1.0 deployed 45 mins ago. HikariCP showing connection pool at 100% utilization. DB team confirms long-running queries from this service. Need to rollback — but this version included a schema migration.",
     "gold_cited_ids": ["KB-00016", "KB-00020"],
     "gold_resolution": "Check if the migration has a down script. If yes: `helm rollback payment-service 0` and run the down migration via DBA. If no down script: evaluate whether the migration was purely additive (added columns/tables). If additive and new columns are empty, rollback is safe — DBA drops the new empty columns. Take thread dump before rollback to capture the long-running query. After rollback, file a hotfix ticket for the query optimization. Per KB-00016, add `leakDetectionThreshold` to prevent recurrence."},

    {"ticket_id": "EVAL-00006", "domain": "identity", "difficulty": "medium", "is_unanswerable": False,
     "title": "svc-analytics API token expired AND AD account locked — both blocking analytics pipeline",
     "description": "Analytics pipeline failing since midnight. Two blockers: (1) API token for external data vendor returning 401 (token was 90 days old), (2) The AD service account svc-analytics is locked from 5 bad password attempts from an old Airflow config.",
     "gold_cited_ids": ["KB-00013", "KB-00012"],
     "gold_resolution": "Fix AD lockout first (blocks more services): `Unlock-ADAccount -Identity svc-analytics`. Check all Airflow connections for cached old password: Airflow UI > Admin > Connections. For the API token: follow the zero-downtime rotation pattern — generate new vendor token, update in Vault, update in the analytics platform config, verify old token is still functional, then revoke old token. Add token expiry monitoring at 75 days per KB-00013."},

    {"ticket_id": "EVAL-00007", "domain": "networking", "difficulty": "hard", "is_unanswerable": False,
     "title": "After NOC failover drill, BGP, OSPF, and DHCP all showing anomalies",
     "description": "Failover drill was conducted where NOC-CORE-01 was taken offline and NOC-CORE-02 activated. Now seeing: BGP session to ISP dropped and not recovering, OSPF routes flapping between 01 and 02, DHCP pool on management VLAN at 98%.",
     "gold_cited_ids": ["KB-00001", "KB-00002", "KB-00006"],
     "gold_resolution": "All three issues stem from the failover. BGP: NOC-CORE-02 may have different BGP config — check hold timers and AS against ISP requirements. Verify ACL permits TCP/179 from 02's IP. OSPF flapping: check if both routers are advertising themselves as the same router-id (common failover mistake). OSPF requires unique router-IDs. Also verify area configs match between 01 and 02. DHCP: the failover may have caused devices to drop and re-request leases without releasing old ones. Clear stale bindings. Check if DHCP failover peer configuration is correctly set to NOC-CORE-02 — if 01's bindings weren't replicated, 02 doesn't know what's allocated."},

    {"ticket_id": "EVAL-00008", "domain": "app_support", "difficulty": "easy", "is_unanswerable": False,
     "title": "Redis running out of memory — session keys with no TTL from new login feature",
     "description": "Redis memory alert: 89% usage, rapidly climbing. New user login session feature was released 3 days ago. Each login creates a session:user:<id> key. There are now 1.3M such keys. None have TTL.",
     "gold_cited_ids": ["KB-00019"],
     "gold_resolution": "Session keys without TTL are filling Redis. Immediate: `redis-cli CONFIG SET maxmemory-policy allkeys-lru` if not set. For the session keys: set TTL on all existing keys: `redis-cli --scan --pattern 'session:user:*' | xargs -L 100 -I{} redis-cli expire {} 86400`. In the application code, update the session creation to use SETEX with TTL=86400 (or appropriate session timeout). Add `spring.session.timeout=3600` if using Spring Session."},

    {"ticket_id": "EVAL-00009", "domain": "identity", "difficulty": "easy", "is_unanswerable": False,
     "title": "svc-deploy-prod locked again — third time this month",
     "description": "svc-deploy-prod is locked again (bad password count 6). Third time this month. Each time it happens we just unlock it. Something is still using the old password. Same scenario as TKT-100006.",
     "gold_cited_ids": ["TKT-100006"],
     "gold_resolution": "Per TKT-100006 root cause, a consumer of this account still has the old password cached. Last time it was Kubernetes imagePullSecret in the prod namespace. Check all consumers systematically: Jenkins credentials, all K8s namespaces (`kubectl get secrets --all-namespaces | grep regcred`), Ansible vault entries, and any CI/CD config files. This is a recurrence — follow the full checklist from TKT-100006 and verify EVERY consumer is updated, not just the obvious ones. Consider converting to a gMSA (Group Managed Service Account) which auto-rotates without this problem."},

    {"ticket_id": "EVAL-00010", "domain": "app_support", "difficulty": "hard", "is_unanswerable": False,
     "title": "Three-service degradation: OOM crashes, connection pool exhaustion, and 504s — same deployment",
     "description": "Deployment of platform v7.0.0 at 10:00 caused: (1) user-service OOMKilled every 30 min (heap dumps show large object), (2) order-service connection pool at 100%, (3) API gateway 504s on /api/v1/recommendations. All three services were updated in v7.0.0.",
     "gold_cited_ids": ["KB-00015", "KB-00016", "KB-00018"],
     "gold_resolution": "Three services, one deployment. user-service OOM: analyze heap dump with Eclipse MAT — likely a new cache or collection introduced in v7.0.0. Increase heap temporarily and add `-XX:+HeapDumpOnOutOfMemoryError`. order-service pool: v7.0.0 may have introduced a query that holds connections — take thread dump to identify the holder. Add `leakDetectionThreshold=60000`. 504s on recommendations: check gateway logs for upstream_response_time — the recommendations endpoint may call user-service (which is OOM crashing) or order-service (pool exhausted), making the recommendation chain slow. Fix the root services first. Consider rolling back v7.0.0 while fixing."},

    # UNANSWERABLE EVAL
    {"ticket_id": "EVAL-00011", "domain": "networking", "difficulty": "medium", "is_unanswerable": True,
     "title": "Juniper SRX HA cluster split-brain after power interruption",
     "description": "Juniper SRX5400 HA cluster entered split-brain state after a PDU power interruption. Both nodes think they are primary. No runbook for Juniper SRX HA recovery exists.",
     "gold_cited_ids": [],
     "gold_resolution": "Escalate to the network security team and open a Juniper TAC case. SRX5400 HA cluster split-brain recovery is not covered in any current KB. Collect: `show chassis cluster status`, `show chassis cluster interfaces`, and system logs from both nodes."},

    {"ticket_id": "EVAL-00012", "domain": "app_support", "difficulty": "hard", "is_unanswerable": True,
     "title": "Distributed tracing showing cross-service timing anomaly — root cause unknown",
     "description": "Jaeger distributed traces show a 400ms unexplained gap between service A calling service B. Both services are healthy. Gap appears only for calls from EU region. No runbook or past precedent.",
     "gold_cited_ids": [],
     "gold_resolution": "Escalate to the observability team and the platform SRE team. This cross-service timing anomaly in EU region is not covered by any current runbook. Collect: specific Jaeger trace IDs with the anomaly, region-specific routing configuration, and network latency measurements between EU pods."},

    {"ticket_id": "EVAL-00013", "domain": "identity", "difficulty": "medium", "is_unanswerable": True,
     "title": "SCIM provisioning to a new internal custom-built app fails with custom error",
     "description": "New internal app 'ProjectTracker' (custom-built by the product team) returns HTTP 422 on all Okta SCIM pushes. The app uses a non-standard SCIM attribute schema. No documentation exists.",
     "gold_cited_ids": [],
     "gold_resolution": "Escalate to the ProjectTracker development team. The non-standard SCIM schema returning 422 errors requires the application developer to fix their SCIM endpoint to handle standard Okta SCIM pushes, or to provide documentation of their custom schema so the Okta connector can be configured. This is not covered in any KB article."},

    {"ticket_id": "EVAL-00014", "domain": "app_support", "difficulty": "easy", "is_unanswerable": False,
     "title": "SSL cert for dev-api.internal expired — developers blocked",
     "description": "dev-api.internal TLS cert expired. All developers using this endpoint for local testing are getting SSL errors. Cert was self-managed (not in IT-PKI inventory). Need to renew.",
     "gold_cited_ids": ["KB-00014"],
     "gold_resolution": "Submit a ServiceNow ticket to IT-PKI under 'Certificate > Internal PKI' with the FQDN dev-api.internal. Since developers are blocked but this is a non-production service, standard 2-day SLA applies (not P1). Generate CSR: `openssl req -new -newkey rsa:2048 -nodes -keyout dev-api.key -out dev-api.csr -subj '/CN=dev-api.internal'`. Include all relevant SANs. After cert arrival, install per the web server type. Add to certificate inventory after installation."},

    {"ticket_id": "EVAL-00015", "domain": "networking", "difficulty": "easy", "is_unanswerable": True,
     "title": "Wi-Fi calling feature on employee phones causes DSCP marking conflicts with UC traffic",
     "description": "After enabling Wi-Fi calling for corporate SIM employees, voice quality on Cisco UC (Jabber) calls degraded significantly. Both Wi-Fi calling and UC traffic are on the same SSID. DSCP conflict suspected. No runbook for this coexistence scenario.",
     "gold_cited_ids": [],
     "gold_resolution": "Escalate to the unified communications and wireless teams jointly. Wi-Fi calling and UC traffic DSCP coexistence is not covered in any current KB or runbook. This requires a wireless QoS policy review that accounts for both carrier Wi-Fi calling DSCP markings and corporate UC policies."},

    {"ticket_id": "EVAL-00016", "domain": "app_support", "difficulty": "medium", "is_unanswerable": False,
     "title": "connection pool exhaustion on report-service — timing matches batch job",
     "description": "report-service timing out on connection acquisition every night at 23:00 exactly. DBA reports long-running queries from report-service at that time. Pattern is new since last week. Looks like the month-end report job recently enabled for nightly runs.",
     "gold_cited_ids": ["KB-00016", "TKT-100014"],
     "gold_resolution": "This is the same root cause as TKT-100014. Nightly batch reports hold connections for the full query duration, exhausting the pool for other requests. From TKT-100014: (1) increase HikariCP `maximumPoolSize` to 20+, (2) Set `statement_timeout=600000` on report queries to prevent runaway holds, (3) Ideally create a separate connection pool for batch/reporting queries with its own size limit so they don't compete with OLTP requests."},

    {"ticket_id": "EVAL-00017", "domain": "identity", "difficulty": "medium", "is_unanswerable": False,
     "title": "User stuck in SAML redirect loop on internal dashboard — third report this week",
     "description": "Three separate users this week have reported a SAML redirect loop on dashboard.corp.example.com. All resolved by clearing cookies but it keeps happening. Need to find the permanent fix.",
     "gold_cited_ids": ["KB-00010", "TKT-100008"],
     "gold_resolution": "Per TKT-100008 root cause: the dashboard application has stale session cookies from the pre-SAML migration (named 'wikisession' or similar). The permanent fix is to invalidate the old cookie name at the application level — have the dev team update the session cookie configuration to use a new cookie name that old clients won't have cached, and set `SameSite=Strict` and an appropriate expiry. The short-term fix of clearing cookies is not sustainable at scale. SAML Tracer on one of the affected users will confirm the redirect loop is InResponseTo-based."},

    {"ticket_id": "EVAL-00018", "domain": "app_support", "difficulty": "easy", "is_unanswerable": False,
     "title": "New pod deployment OOMKilled immediately — memory limit seems correct",
     "description": "notification-worker v3.0 deploying to staging. Each pod gets OOMKilled within 90 seconds. Memory limit is set to 2GB. Developer says the app uses <500MB normally. First time seeing this.",
     "gold_cited_ids": ["KB-00017"],
     "gold_resolution": "Take `kubectl logs <pod> --previous` to see the crash logs. Check Events in `kubectl describe pod` — if OOMKilled, the container exceeded the limit. If the app normally uses <500MB but is hitting 2GB at startup, check for a startup behavior: some applications load a large data file or pre-warm caches on startup that cause a temporary memory spike. Also verify with `kubectl top pod` if the limit is being hit during the first 90 seconds. Consider increasing the limit temporarily to 4GB to get the app running, then profile the startup memory usage to understand the spike."},

    {"ticket_id": "EVAL-00019", "domain": "networking", "difficulty": "medium", "is_unanswerable": False,
     "title": "DNS resolution failing AND DHCP scope issue affecting same floor",
     "description": "All users on floor 6 (VLAN 60) are experiencing two issues: can't resolve internal hostnames, AND 20% of their machines show APIPA addresses. Issues started after the VLAN 60 DHCP scope was relocated to a new server this morning.",
     "gold_cited_ids": ["KB-00003", "KB-00006"],
     "gold_resolution": "Both issues trace to the DHCP server relocation. APIPA (no IP): the new DHCP server may not have the VLAN 60 scope fully configured, or the DHCP helper-address on the VLAN 60 SVI still points to the old server. Check: `show running-config interface vlan 60 | include helper` — update helper-address to new server IP. DNS failure: DHCP option 6 (DNS server) on the new server likely isn't configured with 10.10.1.53. Update DHCP scope option 6 on the new server. After fixing, affected clients need to run `ipconfig /renew` or reconnect."},

    {"ticket_id": "EVAL-00020", "domain": "identity", "difficulty": "hard", "is_unanswerable": False,
     "title": "Post-incident: SSO cert expired, service accounts locked, and SCIM broken — same root as known incident",
     "description": "Following overnight maintenance, three issues discovered at 08:00: SAML SSO failing for all apps (cert-related error), 5 service accounts locked, SCIM push failing for HR system. Similar pattern to INC-0004 and INC-0009.",
     "gold_cited_ids": ["KB-00010", "KB-00012", "INC-0004"],
     "gold_resolution": "Per INC-0004, Okta signing cert expiry causes SSO failures across all SPs simultaneously. Check Okta signing cert in Settings > Keys. If expired, generate new cert and update metadata at all 14+ SP integrations — use the list from the INC-0004 post-mortem. Service account lockouts per INC-0009 pattern: an Okta policy may have changed during maintenance. Check Okta System Log for policy changes and revert if too broad. Unlock accounts in AD. For SCIM: with a cert rotation, SCIM Bearer tokens may need regeneration — check Okta System Log for 401 errors in SCIM push events per KB-00012 for each affected HR system integration."},
]

def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  ✓ {path}: {len(data)} items")

print("Writing corpus files...")
write_json(OUT / "kb.json", KB)
write_json(OUT / "past_tickets.json", PAST_TICKETS)
write_json(OUT / "incidents.json", INCIDENTS)
write_json(OUT / "train_tickets.json", TRAIN_TICKETS)
write_json(OUT / "eval_tickets.json", EVAL_TICKETS)

# Validation: check gold_cited_ids point to real IDs
valid_ids = (
    {a["article_id"] for a in KB} |
    {t["ticket_id"] for t in PAST_TICKETS} |
    {i["incident_id"] for i in INCIDENTS}
)
errors = []
for t in TRAIN_TICKETS + EVAL_TICKETS:
    for cid in t.get("gold_cited_ids", []):
        if cid not in valid_ids:
            errors.append(f"  ✗ {t['ticket_id']} cites unknown ID: {cid}")

if errors:
    print("\nCROSS-REFERENCE ERRORS:")
    for e in errors:
        print(e)
else:
    print(f"\n✓ All {sum(len(t.get('gold_cited_ids',[])) for t in TRAIN_TICKETS + EVAL_TICKETS)} "
          f"gold_cited_ids cross-references valid.")

print(f"\nSummary:")
print(f"  KB articles: {len(KB)}")
print(f"  Past tickets (corpus): {len(PAST_TICKETS)}")
print(f"  Incidents: {len(INCIDENTS)}")
print(f"  Train tickets: {len(TRAIN_TICKETS)} "
      f"({sum(1 for t in TRAIN_TICKETS if not t['is_unanswerable'])} answerable, "
      f"{sum(1 for t in TRAIN_TICKETS if t['is_unanswerable'])} unanswerable)")
print(f"  Eval tickets: {len(EVAL_TICKETS)} "
      f"({sum(1 for t in EVAL_TICKETS if not t['is_unanswerable'])} answerable, "
      f"{sum(1 for t in EVAL_TICKETS if t['is_unanswerable'])} unanswerable)")
