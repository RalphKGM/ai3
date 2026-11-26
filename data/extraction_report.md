# Phase 1 Extraction Report

## Output Files in `data/extracted/`

| File | Path | Size |
|---|---|---|
| `comres_11076_2024.json` | `data/extracted/comres_11076_2024.json` | 343.5 KB |
| `comres_11076_2024.md` | `data/extracted/comres_11076_2024.md` | 172.2 KB |
| `comres_11086_2024.json` | `data/extracted/comres_11086_2024.json` | 106.2 KB |
| `comres_11086_2024.md` | `data/extracted/comres_11086_2024.md` | 67.6 KB |
| `Philippines_1987-en.json` | `data/extracted/Philippines_1987-en.json` | 215.7 KB |
| `Philippines_1987-en.md` | `data/extracted/Philippines_1987-en.md` | 136.9 KB |
| `Philippines_1987-en_cropped.pdf` | `data/extracted/Philippines_1987-en_cropped.pdf` | 313.0 KB |

## Per-Document Extraction Summary

| Document | Total Pages | Extracted | Dropped | Tables | Footnotes | Page-Break Joins | OCR Fixes | QA Warnings |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| `Philippines_1987-en.pdf` | 62 | 51 | 11 ([1, 2, 3]...[61, 62]) | 0 | 0 | 9 | 0 | 0 |
| `comres_11086_2024.pdf` | 31 | 31 | 0 | 8 | 6 | 1 | 140 | 3 |
| `comres_11076_2024.pdf` | 98 | 98 | 0 | 25 | 0 | 3 | 1444 | 3 |

## OCR Repairs & QA Warnings

### `Philippines_1987-en.pdf`
- **OCR fixes applied:** None (native vector text or zero matches)
- **QA warnings:** 0

### `comres_11086_2024.pdf`
- **OCR fixes applied:** `double_space_collapse`=140
- **QA warnings (3 total, showing up to 8):**
  - p.5 [unusual_non_ascii]: ` or make any expenditure Én t SECTION 5. Authorized`
  - p.6 [unusual_non_ascii]: `t and one-half inches (8 ½")in width and fourteen i`
  - p.31 [unusual_non_ascii]: `CONSÚELO B. DIOLA Director IV `

### `comres_11076_2024.pdf`
- **OCR fixes applied:** `glued_CountingMachines`=1, `glued_ARTICLE_roman`=2, `double_space_collapse`=1434, `glued_halfof`=1, `glued_day_month`=1, `glued_SEALINGOF`=1, `glued_ANDS_PPP`=1, `glued_Per_PP_typo`=2, `glued_PDLVoting`=1
- **QA warnings (3 total, showing up to 8):**
  - p.7 [unusual_non_ascii]: `ember's or members' judgmènt, is absent with the in`
  - p.47 [unusual_non_ascii]: ` Turnout Form ("ANNEX F-2″).`
  - p.72 [unusual_non_ascii]: `日 DEPOSITING THE BALLOT IN`

## Verbatim Sample Excerpts

### 1. `Philippines_1987-en.md` — Preamble & Article I
```markdown
## Preamble

We, the sovereign Filipino people, imploring the aid of Almighty God, in order to build a just and humane society and establish a Government that shall embody our ideals and aspirations, promote the common good, conserve and develop our patrimony, and secure to ourselves and our posterity the blessings of independence and democracy under the rule of law and a regime of truth, justice, freedom, love, equality, and peace, do ordain and promulgate this Constitution.

## ARTICLE I. NATIONAL TERRITORY

The national territory comprises the Philippine archipelago, with all the islands and waters embraced therein, and all other territories over which the Philippines has sovereignty or jurisdiction, consisting of its terrestrial, fluvial, and aerial domains, including its territorial sea, the seabed, the subsoil, the insular shelves, and other submarine areas. The waters around, between, and connecting the islands of the archipelago
```

### 2. `Philippines_1987-en.md` — Article V (Suffrage)
```markdown
## ARTICLE V. SUFFRAGE

### Sec 1

Suffrage may be exercised by all citizens of the Philippines not otherwise disqualified by law, who are at least eighteen years of age, and who shall have resided in the Philippines for at least one year and in the place wherein they propose to vote for at least six months immediately preceding the election. No literacy, property, or other substantive requirement shall be imposed on the exercise of suffrage.

### Sec 2

The Congress shall provide a system for securing the secrecy and sanctity of the ballot as well as a system for absentee voting by qualified Filipinos abroad.

The Congress shall also design a procedure for the disabled and the illiterates to vote without the assistance of other persons. Un
```

### 3. `comres_11076_2024.md` — Body Article I Excerpt
```markdown
ARTICLE I DEFINITION OF TERMS

SECTION. 1. Definition of Terms. - As used in this Resolution, the following terms shall mean:

(a) Accessible Polling Place (APP) refers to the venue where the Electoral Board (EB) conducts election-related proceedings for voters who are Persons with Disability and Senior Citizens (SC) and who have previously manifested to cast their votes in an APP. The APP shall be located at the ground floor, preferably near the entrance of the building, free of any physical barriers, and provided with necessary services including assistive devices.

free of any physical barriers, and provided with necessary services including assistive devices.

(b) Accessible Voting Center (AVC) refers to the established voting centers for Indigenous Peoples (IPs) located near or in their communities where they shall cast votes on elec
```

### 4. `comres_11076_2024.md` — Extracted Table Excerpt
```markdown
| CEF NO. | DESCRIPTION | RATE OF DISTRIBUTION PER CLUSTERED PRECINCT | RATE OF DISTRIBUTION PER CLUSTERED PRECINCT |
|-----------------------------------|----------------------------------------------------------------------------------------------------------|-----------------------------------------------|-----------------------------------------------|
| ELECTION FORMS | ELECTION FORMS | ELECTION FORMS | ELECTION FORMS |
| A3 | Poster Indicating Precincts Numbers | 1 | Piece |
| A6 | Official Ballots | 1 | Piece per voter |
| A12 | Paper Seal | 48 | Pieces |
| A13 | Certificate of Votes | 20 | Pieces |
| A14 | Certificate of Receipt of Official Ballots, Other Forms and Supplies by the Electoral Board | 3 | Pieces |
| A27 | Official Receipt of Election Returns | 10 | Pieces |
| ENVELOPES FOR VOTING AND COUNTING | ENVELOPES FOR VOTING AND COUNTING | ENVELOPES FOR VOTING AND COUNTING | ENVELOPES FOR VOTING AND COUNTING |
| A16 | For Rejected ballots, Half of Torn Unused OfficialBallots and Other Half of Torn Unused Official Ballots | 3 | Pieces |
| A17 | For Election Returns | 30 |
```

### 5. `comres_11076_2024.md` — Bilingual Form & Checkboxes Excerpt
```markdown
ANNEX D;"
may wait until a minimum of ten (10) voters have availed themselves of the PPP before obtaining the ballots from their regular precincts;
(e) The absence of an Automated Counting Machine (ACM) in the PPP;
(f) The ballots will be returned to the EB Chairperson of the clustered precincts where they are registered and that their ballots will be fed to the ACM by the EB Chairperson; and
(g) Such other relevant information.

SEC. 51. Remuneration and Other Benefits.- The PPP/S-PPP SS shall have the same honoraria and benefits as the regular EB SS mentioned in the provision for honoraria and allowances as stated in Section 13 (a) of this Resolution.

SEC. 52. Election Forms, Documents, and Supplies for Voting in PPPs and S-PPPs. – Except when authorized to do so earlier by the Commission, the PPP SS/S-PPP SS shall obtain the following forms, documents, and supplies early in the morni
```

### 6. `comres_11086_2024.md` — Resolution Body Excerpt
```markdown
Resolution No. 11086 (IRR of the Fair Election Act, RA 9006)

<!-- Source: comres_11086_2024.pdf | Pages: 31 | Scope: 12 May 2025 National, Local, and BARMM Parliamentary Elections | Remediated pages: 12 -->

### REPUBLIC OF THE PHILIPPINES COMMISSION ON ELECTIONS Intramuros, Manila

RULES AND REGULATIONS IMPLEMENTING REPUBLIC ACT NO. 9006, OTHERWISE KNOWN AS THE "FAIR ELECTION ACT", IN CONNECTION WITH THE 12 MAY 2025 NATIONAL, LOCAL, & BANGSAMOROPARLIAMENTARY ELECTIONS.

GARCIA, George Erwin M., Chairman INTING, Socorro B., Commissioner CASQUEJO, Marlon S., Commissioner FEROLINO, Aimee P., Commissioner BULAY, Rey E., Commissioner MACEDA, JR., Ernesto Ferdinand P., Commissioner CELIS, Nelson J., Commissioner

Promulgated: 09 December 2024

WHEREAS, Republic Act No. 9006, otherwise known as the "Fair Election Act" provides for the holding of free, orderly, honest, peaceful and credible el
```

### 7. `comres_11086_2024.md` — Table & Footnotes Excerpt
```markdown
| For Candidates/ Registered Political Parties for a National Elective Position | Not more than a total of one hundred twenty (120) minutes of television advertisement, on a per station basis, whether appearing on national, regional, or local, free or cable television, and one hundred eighty (180) minutes of radio advertising, on a per station basis, whether airing on national, regional, or local radio, whether by purchase or donation. |
|---------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```

```markdown
## Footnotes

- **[p. 1]** i RA 7166, entitled "AN ACT PROVIDING FOR SYNCHRONIZED NATIONAL AND LOCAL ELECTIONS AND FOR ELECTORAL REFORMS, AUTHORIZING AND APPROPRIATIONS THEREFOR, AND FOR OTHER PURPOSES".
- **[p. 2]** 2 An Act Introducing Additional Reforms In The Electoral System And For Other Purposes
- **[p. 2]** 3 Supra At Note 1
- **[p. 9]** Republic Act No. 9710, also known as "THE MAGNA CARTA OF WOMEN".
- **[p. 17]** Republic Act No. 11207 entitled "AN ACT PROVIDING FOR REASONABLE RATES FOR POLITICAL ADVERTISEMENTS, AMENDING FOR THE PURPOSE SECTION 11 OF REPUBLIC ACT NO. 9006, OTHERWISE KNOWN AS THE FAIR ELECTION ACT". COMELEC Resolution No. 10517.
- **[p. 19]** 6 RA 3571, entitled "AN ACT TO PROHIBIT THE CUTTING, DESTROYING OR INJURI
```
