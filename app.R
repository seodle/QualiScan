# ══════════════════════════════════════════════════════════════════════════════
# QualiScan — Extraction de texte manuscrit depuis PDF (Interface Shiny)
# Utilise l'API Infomaniak (OpenAI-compatible) avec un modèle vision
# ══════════════════════════════════════════════════════════════════════════════

# ── Packages requis ────────────────────────────────────────────────────────────
required_packages <- c(
  "shiny", "bslib", "bsicons",
  "pdftools", "httr2",
  "base64enc", "glue", "markdown", "shinyjs", "png", "jpeg", "rmarkdown"
)

missing_pkgs <- required_packages[!sapply(required_packages, requireNamespace, quietly = TRUE)]
if (length(missing_pkgs) > 0) {
  message("Installation des packages manquants : ", paste(missing_pkgs, collapse = ", "))
  install.packages(missing_pkgs, repos = "https://cloud.r-project.org")
}

library(shiny)
library(bslib)

# Timeout étendu pour la génération de PDF (rmarkdown + xelatex)
options(shiny.downloadHandler.timeout = 300)

# Ajouter pandoc (/opt/homebrew/bin) et xelatex (/Library/TeX/texbin) au PATH
local({
  extra <- c("/opt/homebrew/bin", "/Library/TeX/texbin")
  cur   <- strsplit(Sys.getenv("PATH"), ":")[[1]]
  add   <- setdiff(extra, cur)
  if (length(add)) Sys.setenv(PATH = paste(c(Sys.getenv("PATH"), add), collapse = ":"))
})
# Indiquer à rmarkdown où se trouve pandoc
rmarkdown::find_pandoc(dir = "/opt/homebrew/bin")
library(bsicons)
library(pdftools)
library(httr2)
library(base64enc)
library(glue)
library(markdown)
library(shinyjs)

# ── Chargement des credentials depuis .env ─────────────────────────────────────
env_file <- file.path(getwd(), ".env")
if (file.exists(env_file)) {
  readRenviron(env_file)
  message(".env chargé depuis : ", env_file)
} else {
  message("Aucun fichier .env trouvé — les credentials devront être saisis manuellement.")
}
ENV_API_KEY    <- Sys.getenv("api_key",    unset = "")
ENV_PRODUCT_ID <- Sys.getenv("product_id", unset = "")
ENV_LOADED     <- nzchar(ENV_API_KEY) && nzchar(ENV_PRODUCT_ID)

# Dossier de sauvegarde des sessions (créé si absent)
SAVE_DIR <- file.path(getwd(), "sauvegarde")
dir.create(SAVE_DIR, recursive = TRUE, showWarnings = FALSE)

# ── Prompt d'extraction ────────────────────────────────────────────────────────
EXTRACTION_PROMPT <- 'Vous êtes un expert en OCR. Votre unique rôle est de TRANSCRIRE exactement ce qui est écrit à la main ou coché sur les images de ce document. Vous n\'inventez rien. Vous ne complétez pas. Vous ne déduisez pas.

FORMAT DE SORTIE OBLIGATOIRE — Markdown strict, structure exacte ci-dessous :

## Remarques générales

[Texte manuscrit copié mot pour mot. Abréviations conservées telles quelles : é., CE, ex., adj., …]

## Critères qualitatifs — 1ère partie

| Critère d\'évaluation | Oui/Non | N° exercices | Observations / Commentaires |
|---|:---:|:---:|---|
| [texte imprimé du critère] | [voir règle cases ci-dessous] | [chiffres visibles] | [texte manuscrit, mot pour mot] |

## Critères qualitatifs — 2ème partie

| Critère d\'évaluation | Oui/Non | N° exercices | Observations / Commentaires |
|---|:---:|:---:|---|
| [texte imprimé du critère] | [voir règle cases ci-dessous] | [chiffres visibles] | [texte manuscrit, mot pour mot] |

## Améliorations nécessaires

[Regardez physiquement la zone réponse sous la question imprimée "Selon vous, des améliorations sont-elles nécessaires avant la généralisation de l\'évaluation ?". Copiez LETTRE PAR LETTRE ce qui y est écrit à la main. Cette réponse est unique et personnelle : elle ne peut pas être devinée ni reconstituée. Si vous ne pouvez pas lire un mot clairement, écrivez [mot?]. Si la zone est vide ou illisible, écrivez *[Section non visible]*. N\'écrivez JAMAIS une réponse plausible ou générique à la place du texte réel.]

RÈGLES ABSOLUES — toute violation est une erreur grave :

CASES À COCHER — lisez chaque case individuellement et de façon indépendante :
- La colonne "Oui/Non" correspond à deux cases distinctes sur le document : une case "Oui" et une case "Non".
- Si la case "Oui" est cochée (✓, ×, trait, gribouillage) → écrivez "Oui".
- Si la case "Non" est cochée (✓, ×, trait, gribouillage) → écrivez "Non".
- Si AUCUNE des deux cases n\'est cochée → laissez la cellule vide.
- Ne déduisez JAMAIS l\'état d\'une case à partir du contexte ou des autres cases. Chaque case est indépendante.
- Une case non cochée n\'est PAS un "Non" : c\'est une case vide. Respectez cette distinction.

TRANSCRIPTION PURE — règle fondamentale :
- Recopiez uniquement ce qui est physiquement visible dans CHAQUE zone délimitée. N\'ajoutez, n\'inventez, ni ne déduisez AUCUNE information.
- INTERDICTION ABSOLUE de générer une réponse "qui pourrait convenir" à la place du texte réel. Toute réponse inventée est une faute grave.
- Si vous hésitez entre ce que vous voyez et ce qui serait logique : écrivez ce que vous VOYEZ, pas ce qui semble logique.
- Si un mot est incertain : [mot?]. Si une section est illisible : *[Section non visible]*.
- Ne faites pas migrer du texte d\'une section à une autre.
- Abréviations : reproduisez-les exactement (é., adj., CE…), ne les développez jamais.
- Utilisez `##` pour les titres de section.
- Ne mentionnez jamais la discipline, le type d\'évaluation, ni la section.
- Aucune phrase d\'introduction, de conclusion ou de commentaire de votre part.'

VISION_MODELS <- c("Qwen/Qwen3.5-122B-A10B-FP8")

# ── Fonctions helpers ──────────────────────────────────────────────────────────

#' Redimensionne et compresse une image (raw bytes) via magick
#' Retourne des bytes JPEG compressés
compress_image_bytes <- function(img_bytes, mime_type, max_dim = 1600,
                                 quality = 82L) {
  if (!requireNamespace("magick", quietly = TRUE))
    return(list(bytes = img_bytes, mime = mime_type))

  tmp_in  <- tempfile(fileext = if (grepl("jpeg", mime_type)) ".jpg" else ".png")
  tmp_out <- tempfile(fileext = ".jpg")
  on.exit({
    if (file.exists(tmp_in))  file.remove(tmp_in)
    if (file.exists(tmp_out)) file.remove(tmp_out)
  }, add = TRUE)

  writeBin(img_bytes, tmp_in)

  img <- tryCatch(magick::image_read(tmp_in), error = function(e) NULL)
  if (is.null(img)) return(list(bytes = img_bytes, mime = mime_type))

  info <- magick::image_info(img)
  if (max(info$width, info$height) > max_dim) {
    img <- magick::image_resize(img, paste0(max_dim, "x", max_dim, ">"))
  }

  img <- magick::image_convert(img, format = "jpeg")
  magick::image_write(img, tmp_out, format = "jpeg", quality = quality)

  out_bytes <- readBin(tmp_out, "raw", file.size(tmp_out))
  list(bytes = out_bytes, mime = "image/jpeg")
}

#' Extrait tous les JPEG embarqués depuis les octets bruts d'un PDF
#' (utilisé quand pdf_convert produit une image blanche)
extract_jpegs_from_pdf_raw <- function(pdf_path) {
  raw <- readBin(pdf_path, "raw", file.size(pdf_path))
  n   <- length(raw)

  # Positions des SOI JPEG : FF D8 FF
  starts <- which(
    raw[seq_len(n - 2)]       == as.raw(0xFF) &
    raw[seq_len(n - 2) + 1]   == as.raw(0xD8) &
    raw[seq_len(n - 2) + 2]   == as.raw(0xFF)
  )

  if (length(starts) == 0) return(list())

  jpegs <- vector("list", length(starts))
  for (i in seq_along(starts)) {
    s <- starts[i]
    e <- if (i < length(starts)) starts[i + 1] - 1L else n

    segment <- raw[s:e]
    ns      <- length(segment)

    # Dernier EOI (FF D9) du segment
    eoi <- which(
      segment[seq_len(ns - 1)]     == as.raw(0xFF) &
      segment[seq_len(ns - 1) + 1] == as.raw(0xD9)
    )

    if (length(eoi) > 0) {
      jpegs[[i]] <- segment[seq_len(tail(eoi, 1) + 1L)]
    }
  }

  Filter(Negate(is.null), jpegs)
}

#' Retourne l'image d'une page PDF en base64
#' — essaie le rendu standard ; si blanc, extrait le JPEG embarqué
pdf_page_to_base64 <- function(pdf_path, page_num, dpi = 200,
                               cached_jpegs = NULL) {

  # ── Priorité 1 : JPEG embarqué (évite tout appel à poppler/pdf_convert) ──
  # pdf_convert peut provoquer un segfault sur certains PDFs scannés ;
  # si des JPEG ont déjà été extraits du PDF, on les utilise directement.
  jpegs <- if (!is.null(cached_jpegs)) cached_jpegs
           else extract_jpegs_from_pdf_raw(pdf_path)

  if (length(jpegs) >= page_num) {
    return(list(b64 = base64enc::base64encode(jpegs[[page_num]]), mime = "image/jpeg"))
  }
  # Si la page n'a pas de JPEG embarqué, on tombe sur pdf_convert ci-dessous

  # ── Priorité 2 : rendu PNG via poppler (PDFs vectoriels sans images) ──────
  tmp_png <- tempfile(fileext = ".png")
  on.exit(if (file.exists(tmp_png)) file.remove(tmp_png), add = TRUE)

  suppressWarnings(
    pdftools::pdf_convert(pdf_path, format = "png", pages = page_num,
                          filenames = tmp_png, dpi = dpi, verbose = FALSE)
  )

  if (file.exists(tmp_png) && file.size(tmp_png) > 0) {
    img_bytes <- readBin(tmp_png, "raw", file.info(tmp_png)$size)
    return(list(b64 = base64enc::base64encode(img_bytes), mime = "image/png"))
  }

  stop("Impossible d'extraire l'image de la page ", page_num)
}

#' Appelle l'API Infomaniak avec une image base64 et renvoie le texte extrait
#' img_list : list of list(b64, mime) — une entrée par page
call_infomaniak_api <- function(img_list, api_key, product_id, model_name,
                                prompt, timeout_sec = 300) {
  api_url <- glue::glue(
    "https://api.infomaniak.com/2/ai/{product_id}/openai/v1/chat/completions"
  )

  # Construire le contenu : texte du prompt + une image par page
  image_blocks <- lapply(img_list, function(img) {
    list(
      type      = "image_url",
      image_url = list(url = paste0("data:", img$mime, ";base64,", img$b64))
    )
  })

  content_blocks <- c(
    list(list(type = "text", text = prompt)),
    image_blocks
  )

  payload <- list(
    model    = model_name,
    messages = list(
      list(role = "user", content = content_blocks)
    ),
    max_completion_tokens    = 4000,
    temperature              = 0.1,
    # Désactive le mode "thinking" de Qwen3 pour des réponses immédiates
    chat_template_kwargs     = list(enable_thinking = FALSE)
  )

  # req_error(is_error = ...) empêche httr2 de lever une erreur automatique
  # sur les 4xx/5xx afin qu'on puisse lire le corps d'erreur de l'API
  resp <- request(api_url) |>
    req_headers(
      "Authorization" = paste("Bearer", api_key),
      "Content-Type"  = "application/json"
    ) |>
    req_body_json(payload) |>
    req_method("POST") |>
    req_timeout(timeout_sec) |>
    req_error(is_error = function(r) FALSE) |>
    req_perform()

  if (resp_status(resp) != 200) {
    body_err <- tryCatch(resp_body_string(resp), error = function(e) "—")
    stop(paste0("HTTP ", resp_status(resp), " — ", substr(body_err, 1, 600)))
  }

  resp_json <- resp_body_json(resp)

  if (!is.null(resp_json$choices) && length(resp_json$choices) > 0) {
    msg <- resp_json$choices[[1]]$message

    # Extraire le contenu (peut être une chaîne ou une liste pour les modèles vision)
    content <- msg$content
    if (is.list(content)) {
      # Format multimodal : liste de blocs {type, text}
      text_blocks <- Filter(function(b) identical(b$type, "text"), content)
      content <- paste(sapply(text_blocks, `[[`, "text"), collapse = "\n")
    }

    # Fallback : reasoning_content (Qwen3 thinking mode)
    if (!nzchar(trimws(content %||% ""))) {
      content <- msg$reasoning_content %||% ""
    }

    # Supprimer les balises <think>…</think> si présentes
    content <- gsub("<think>[\\s\\S]*?</think>", "", content %||% "", perl = TRUE)
    content <- trimws(content)

    if (nzchar(content)) return(content)
  }

  stop("Réponse API vide ou inattendue.")
}

#' Extrait les métadonnées depuis le nom du fichier PDF
#' Format attendu : NOM_Prénom_TYPE_Discipline_Section_Etablissement.pdf
parse_pdf_filename <- function(filename) {
  base  <- tools::file_path_sans_ext(basename(filename))
  parts <- strsplit(base, "_")[[1]]

  type_tokens <- c("EVACOM", "TAF", "EC")
  type_idx    <- which(parts %in% type_tokens)

  if (length(type_idx) == 0) {
    return(list(
      nom = base, prenom = "", teacher_name = base,
      type_epreuve = "", discipline = "", section = "", etablissement = ""
    ))
  }

  type_idx     <- type_idx[1]
  prenom       <- if (type_idx >= 2) parts[type_idx - 1]                            else ""
  nom          <- if (type_idx >= 3) paste(parts[seq_len(type_idx - 2)], collapse = " ") else parts[1]
  type_epreuve <- parts[type_idx]
  discipline   <- if (length(parts) >= type_idx + 1) parts[type_idx + 1] else ""
  section      <- if (length(parts) >= type_idx + 2) parts[type_idx + 2] else ""
  etablissement <- if (length(parts) >= type_idx + 3) parts[type_idx + 3] else ""

  list(
    nom           = nom,
    prenom        = prenom,
    teacher_name  = paste(trimws(nom), trimws(prenom)),
    type_epreuve  = type_epreuve,
    discipline    = discipline,
    section       = section,
    etablissement = etablissement
  )
}

#' Formate le bloc Markdown final pour un enseignant
build_teacher_markdown <- function(teacher_name, discipline, type_epreuve,
                                   section, etablissement, result) {
  meta_lines <- c(
    if (nzchar(discipline))    paste0("**Discipline :** ",     discipline),
    if (nzchar(type_epreuve))  paste0("**Type d'épreuve :** ", type_epreuve),
    if (nzchar(section))       paste0("**Section :** ",        section),
    if (nzchar(etablissement)) paste0("**Établissement :** ",  etablissement)
  )
  meta_block <- if (length(meta_lines) > 0) paste(meta_lines, collapse = "  \n") else ""

  paste0(
    "# Extraction — ", teacher_name, "\n\n",
    if (nzchar(meta_block)) paste0(meta_block, "\n\n") else "",
    "_Généré le ", format(Sys.time(), "%d/%m/%Y à %H:%M:%S"), "_\n\n",
    "---\n\n",
    result
  )
}

# ── Générateur de rapport PDF (Rmd + xelatex) ────────────────────────────────

# Fixe les largeurs de colonnes dans les tableaux Markdown 4 colonnes :
# donne ~50 % de l'espace à la colonne Observations.
# Décale tous les ## (et plus profonds) d'un niveau dans le contenu IA :
#   ## → ###  (subsection → subsubsection)
#   ### → #### etc.
# Cela libère ## pour les noms d'enseignants et # pour les groupes.
downshift_headings <- function(md) {
  gsub("(?m)^(#{2,})", "#\\1", md, perl = TRUE)
}

fix_table_widths <- function(md) {
  lines <- strsplit(md, "\n")[[1]]
  for (i in seq_along(lines)) {
    ln <- lines[i]
    # Largeurs colonnes pour tableaux 4 colonnes (Critère | O/N | N° | Observations)
    if (grepl("^[|][ |:~-]+[|]$", ln) && grepl("-", ln, fixed = TRUE) &&
        !grepl("[a-zA-Z0-9]", ln)) {
      inner <- gsub("^\\||\\|$", "", ln)
      cols  <- strsplit(inner, "\\|")[[1]]
      if (length(cols) == 4) {
        lines[i] <- "| :------------------------- | :------: | :----: | :---------------------------------------------------- |"
      }
    }
    # Saut de page avant la 2ème partie (après downshift, le titre est en ###)
    if (grepl("^#{2,}\\s+.*2", ln, ignore.case = TRUE) &&
        grepl("(2.me|deuxi|partie)", ln, ignore.case = TRUE)) {
      lines[i] <- paste0("\\newpage\n\n", ln)
    }
  }
  paste(lines, collapse = "\n")
}

generate_rmd_report <- function(all_results,
                                title = "Analyse qualitative des pr\u00e9tests") {
  if (length(all_results) == 0) return("")

  get_key <- function(e) {
    paste(e$discipline %||% "", e$type_epreuve %||% "", e$section %||% "",
          sep = "\u001F")
  }
  get_label <- function(key) {
    parts <- strsplit(key, "\u001F")[[1]]
    paste(Filter(nzchar, parts), collapse = " \u00b7 ")
  }

  keys   <- vapply(all_results, get_key, character(1))
  groups <- split(all_results, keys)
  groups <- groups[sort(names(groups))]

  n_ens  <- length(all_results)
  n_grps <- length(groups)

  safe_title <- gsub("'", "''", title)
  style_path <- gsub("\\\\", "/", file.path(getwd(), "report_style.tex"))

  yaml <- paste0(
    "---\n",
    "title: '", safe_title, "'\n",
    "date: ''\n",
    "lang: fr\n",
    "output:\n",
    "  pdf_document:\n",
    "    latex_engine: xelatex\n",
    "    toc: false\n",
    "    number_sections: false\n",
    "    keep_tex: false\n",
    "    includes:\n",
    "      in_header: '", style_path, "'\n",
    "geometry: 'a4paper, top=2cm, bottom=2.5cm, left=2cm, right=2cm'\n",
    "fontsize: 10pt\n",
    "---\n\n"
  )

  intro <- ""

  # Sections : chaque enseignant commence sur une nouvelle page
  sections <- paste(
    mapply(function(key, i) {
      lbl     <- get_label(key)
      entries <- groups[[key]]
      n       <- length(entries)

      teacher_blocks <- paste(
        mapply(function(e, j) {
          etab    <- if (nzchar(e$etablissement %||% ""))
            paste0(" *(", e$etablissement, ")*") else ""
          # Décale les ## du contenu IA en ### pour la hiérarchie correcte
          content <- downshift_headings(
            fix_table_widths(e$raw_content %||% "*Aucun contenu extrait.*")
          )
          page_break <- if (j > 1) "\\newpage\n\n" else ""
          paste0(page_break, "## ", e$teacher_name, etab, "\n\n", content, "\n\n")
        }, entries, seq_along(entries), SIMPLIFY = FALSE),
        collapse = "\n"
      )

      # Saut de page avant chaque groupe (sauf le tout premier)
      grp_break <- if (i > 1) "\\newpage\n\n" else ""
      paste0(
        grp_break,
        "# ", lbl, "\n\n",
        "*", n, " enseignant", if (n > 1) "s" else "", "*\n\n",
        teacher_blocks
      )
    }, names(groups), seq_along(groups)),
    collapse = "\n"
  )

  paste0(yaml, intro, sections)
}

# Construit le modal d'édition Markdown pour un enseignant
edit_modal <- function(teacher_name, raw_content) {
  modalDialog(
    title = tags$span(bs_icon("pencil-square"), " Modifier — ", teacher_name),
    size  = "xl",
    easyClose = FALSE,
    tags$p(
      class = "text-muted small mb-2",
      "Éditez le Markdown directement. Les tableaux utilisent le format ",
      tags$code("| col | col |"), ". Sauvegardez pour mettre à jour la session et le rapport."
    ),
    textAreaInput(
      "edit_textarea",
      label  = NULL,
      value  = raw_content,
      width  = "100%",
      height = "65vh",
      resize = "vertical"
    ),
    tags$style("#edit_textarea { font-family: monospace; font-size: .85rem; }"),
    footer = tagList(
      actionButton("edit_save",   tags$span(bs_icon("floppy2"), " Sauvegarder"),
                   class = "btn-primary"),
      actionButton("edit_cancel", "Annuler", class = "btn-outline-secondary ms-2")
    )
  )
}

# ── UI ─────────────────────────────────────────────────────────────────────────
ui <- page_sidebar(
  title        = tags$span(bs_icon("file-earmark-text"), " QualiScan"),
  window_title = "QualiScan",
  theme = bs_theme(
    bootswatch  = "cosmo",
    primary     = "#0D6EFD",
    base_font   = font_google("Inter"),
    heading_font = font_google("Inter")
  ),
  useShinyjs(),

  # ── Barre latérale ────────────────────────────────────────────────────────────
  sidebar = sidebar(
    width = 320,
    open  = "always",

    accordion(
      open = c("panel_api", "panel_doc"),

      accordion_panel(
        title = tags$span(bs_icon("key"), " Connexion API"),
        value = "panel_api",

        if (ENV_LOADED) {
          div(
            class = "alert alert-success d-flex align-items-center gap-2 py-2 px-3 mb-3",
            style = "font-size:.85rem;",
            bs_icon("check-circle-fill"),
            "Credentials chargés depuis", tags$code(".env")
          )
        },

        passwordInput(
          "api_key",
          tags$span(bs_icon("shield-lock"), " Clé API Infomaniak"),
          value       = ENV_API_KEY,
          placeholder = "Votre clé Bearer"
        ),
        textInput(
          "product_id",
          tags$span(bs_icon("cpu"), " Product ID"),
          value       = ENV_PRODUCT_ID,
          placeholder = "ex : 12345"
        ),
        selectizeInput(
          "model_name",
          tags$span(bs_icon("robot"), " Modèle vision"),
          choices  = VISION_MODELS,
          selected = VISION_MODELS[1],
          options  = list(create = TRUE, placeholder = "Choisir ou saisir…")
        )
      ),

      accordion_panel(
        title = tags$span(bs_icon("file-earmark-pdf"), " Document"),
        value = "panel_doc",

        fileInput(
          "pdf_file",
          tags$span(bs_icon("upload"), " Fichier PDF"),
          accept        = ".pdf",
          buttonLabel   = "Choisir…",
          placeholder   = "Aucun fichier"
        ),

        uiOutput("parsed_badge"),

        textInput(
          "teacher_name",
          tags$span(bs_icon("person"), " Enseignant (NOM Prénom)"),
          placeholder = "Rempli automatiquement"
        ),
        fluidRow(
          column(6, textInput(
            "discipline",
            tags$span(bs_icon("book"), " Discipline"),
            placeholder = "Rempli auto."
          )),
          column(6, textInput(
            "type_epreuve",
            tags$span(bs_icon("clipboard"), " Type"),
            placeholder = "EVACOM…"
          ))
        ),
        fluidRow(
          column(6, textInput(
            "section",
            tags$span(bs_icon("diagram-3"), " Section"),
            placeholder = "CT / LC / LS"
          )),
          column(6, textInput(
            "etablissement",
            tags$span(bs_icon("building"), " Établissement"),
            placeholder = "Rempli auto."
          ))
        ),
        sliderInput(
          "dpi",
          tags$span(bs_icon("aspect-ratio"), " Résolution (DPI)"),
          min = 100, max = 300, value = 200, step = 25,
          ticks = FALSE
        ),
        numericInput(
          "timeout_sec",
          tags$span(bs_icon("clock"), " Timeout API (secondes)"),
          value = 300, min = 60, max = 600, step = 30
        )
      )
    ),

    hr(),

    actionButton(
      "process_btn",
      tags$span(bs_icon("play-circle"), " Lancer l'extraction"),
      class = "btn-primary w-100 fw-bold",
      style = "font-size: 1.05rem;"
    ),

    br(),

    hr(),

    # ── Sauvegarde / restauration de session ──────────────────────────────────
    div(
      class = "small text-muted fw-semibold mb-1",
      bs_icon("floppy"), " Sauvegarde de session"
    ),
    tags$small(
      class = "text-muted d-block mb-2",
      style = "word-break:break-all;",
      paste0("→ sauvegarde/")
    ),
    conditionalPanel(
      condition = "output.has_results",
      actionButton(
        "save_session",
        tags$span(bs_icon("floppy2"), " Sauvegarder"),
        class = "btn-warning w-100 mb-2"
      )
    ),
    uiOutput("load_session_ui"),
    actionButton(
      "load_session_btn",
      tags$span(bs_icon("folder2-open"), " Charger cette session"),
      class = "btn-outline-secondary w-100"
    )
  ),

  # ── Contenu principal ─────────────────────────────────────────────────────────
  navset_card_tab(
    id = "main_tabs",

    nav_panel(
      title = tags$span(bs_icon("card-text"), " Résultats"),
      value = "tab_results",

      uiOutput("results_ui")
    ),

    nav_panel(
      title = tags$span(bs_icon("list-check"), " Documents déjà traités"),
      value = "tab_session",

      uiOutput("session_ui")
    ),

    nav_panel(
      title = tags$span(bs_icon("file-earmark-richtext"), " Rapport"),
      value = "tab_rapport",

      uiOutput("rapport_ui")
    ),

    nav_panel(
      title = tags$span(bs_icon("terminal"), " Journal"),
      value = "tab_log",

      uiOutput("log_output")
    )
  )
)

# ── Server ─────────────────────────────────────────────────────────────────────
server <- function(input, output, session) {

  rv <- reactiveValues(
    log         = character(0),
    current_md  = NULL,        # Markdown complet du dernier traitement (avec en-tête)
    current_teacher = NULL,    # Nom du dernier enseignant traité
    edit_teacher    = NULL,    # Enseignant en cours d'édition
    all_results = list(),      # list[teacher] = list(teacher_name, discipline, type_epreuve,
                               #                      section, etablissement, raw_content, full_md)
    processing  = FALSE
  )

  # ── Parsing automatique au chargement du PDF ──────────────────────────────────
  observeEvent(input$pdf_file, {
    req(input$pdf_file)
    info <- parse_pdf_filename(input$pdf_file$name)

    updateTextInput(session, "teacher_name",  value = info$teacher_name)
    updateTextInput(session, "discipline",    value = info$discipline)
    updateTextInput(session, "type_epreuve",  value = info$type_epreuve)
    updateTextInput(session, "section",       value = info$section)
    updateTextInput(session, "etablissement", value = info$etablissement)
  })

  output$parsed_badge <- renderUI({
    req(input$pdf_file)
    info <- parse_pdf_filename(input$pdf_file$name)
    if (!nzchar(info$type_epreuve)) return(NULL)

    label <- paste0(
      info$type_epreuve,
      if (nzchar(info$section)) paste0(" · ", info$section) else "",
      if (nzchar(info$discipline)) paste0(" · ", info$discipline) else ""
    )

    div(
      class = "alert alert-info d-flex align-items-center gap-2 py-2 px-3 mb-2",
      style = "font-size:.8rem;",
      bs_icon("magic"),
      tags$span("Extrait du nom de fichier : ", tags$strong(label))
    )
  })

  # ── Helpers internes ──────────────────────────────────────────────────────────
  add_log <- function(msg, level = "INFO") {
    ts   <- format(Sys.time(), "%H:%M:%S")
    icon <- switch(level,
      "OK"    = "✓",
      "ERROR" = "✗",
      "WARN"  = "⚠",
      "→"
    )
    entry <- paste0("[", ts, "] ", icon, " ", msg)
    rv$log <- c(rv$log, entry)
  }

  # ── Extraction principale ─────────────────────────────────────────────────────
  observeEvent(input$process_btn, {

    if (rv$processing) {
      showNotification("Un traitement est déjà en cours.", type = "warning")
      return()
    }

    # Validation des champs obligatoires
    errs <- character(0)
    if (!nzchar(trimws(input$api_key)))      errs <- c(errs, "Clé API manquante")
    if (!nzchar(trimws(input$product_id)))   errs <- c(errs, "Product ID manquant")
    if (!nzchar(trimws(input$teacher_name))) errs <- c(errs, "Nom de l'enseignant manquant")
    if (is.null(input$pdf_file))             errs <- c(errs, "Aucun PDF sélectionné")

    if (length(errs) > 0) {
      showNotification(
        paste("Champs manquants :", paste(errs, collapse = " | ")),
        type = "error", duration = 6
      )
      return()
    }

    rv$processing   <- TRUE
    rv$current_md   <- NULL
    disable("process_btn")

    teacher       <- trimws(input$teacher_name)
    discipline    <- trimws(input$discipline)
    type_epreuve  <- trimws(input$type_epreuve)
    section       <- trimws(input$section)
    etablissement <- trimws(input$etablissement)
    pdf_path      <- input$pdf_file$datapath

    meta_str <- paste(Filter(nzchar, c(discipline, type_epreuve, section, etablissement)), collapse = " · ")
    add_log(paste0("=== Début traitement : ", teacher, " ==="))
    if (nzchar(meta_str)) add_log(paste("Métadonnées :", meta_str))
    add_log(paste("Modèle :", input$model_name))

    tryCatch({

      n_pages <- pdftools::pdf_info(pdf_path)$pages
      add_log(paste(n_pages, "page(s) détectée(s) dans le PDF"))

      # Pré-extraction des JPEG embarqués (pour les PDF scannés rendus blancs)
      add_log("Analyse du PDF (détection images embarquées)…")
      cached_jpegs <- extract_jpegs_from_pdf_raw(pdf_path)
      if (length(cached_jpegs) > 0) {
        add_log(paste0(length(cached_jpegs), " image(s) JPEG embarquée(s) détectée(s)"), "OK")
      }

      # ── Phase 1 : extraction et compression de toutes les images ──────────────
      img_list <- list()

      withProgress(
        message = paste0("Traitement : ", teacher),
        value   = 0,
        {
          incProgress(0, detail = "Extraction des images…")

          for (p in seq_len(n_pages)) {
            add_log(paste0("Page ", p, "/", n_pages, " — extraction image…"))

            img_info <- tryCatch(
              pdf_page_to_base64(pdf_path, p, dpi = input$dpi,
                                 cached_jpegs = cached_jpegs),
              error = function(e) {
                add_log(paste0("Erreur image page ", p, " : ", e$message), "ERROR")
                NULL
              }
            )
            if (is.null(img_info)) next

            raw_bytes <- base64enc::base64decode(img_info$b64)
            compressed <- tryCatch(
              compress_image_bytes(raw_bytes, img_info$mime,
                                   max_dim = 1200L, quality = 75L),
              error = function(e) {
                add_log(paste0("Compression page ", p, " ignorée : ", e$message), "WARN")
                list(bytes = raw_bytes, mime = img_info$mime)
              }
            )

            img_kb_orig <- round(length(raw_bytes) / 1024, 1)
            img_kb_comp <- round(length(compressed$bytes) / 1024, 1)
            add_log(paste0("Page ", p, " — ", img_kb_orig, " Ko → ", img_kb_comp, " Ko"), "OK")

            img_list[[length(img_list) + 1]] <- list(
              b64  = base64enc::base64encode(compressed$bytes),
              mime = compressed$mime
            )

            incProgress(0.5 / n_pages)
          }

          # ── Phase 2 : un seul appel API avec toutes les images ─────────────
          if (length(img_list) == 0) {
            add_log("Aucune image extraite.", "ERROR")
          } else {
            total_kb <- round(sum(sapply(img_list, function(i) nchar(i$b64) * 3/4 / 1024)), 0)
            add_log(paste0("Appel API — ", length(img_list), " image(s), ",
                           total_kb, " Ko total…"))

            t0 <- proc.time()["elapsed"]

            api_result <- tryCatch(
              call_infomaniak_api(
                img_list    = img_list,
                api_key     = input$api_key,
                product_id  = input$product_id,
                model_name  = input$model_name,
                prompt      = EXTRACTION_PROMPT,
                timeout_sec = input$timeout_sec
              ),
              error = function(e) {
                add_log(e$message, "ERROR")
                NULL
              }
            )

            elapsed <- round(proc.time()["elapsed"] - t0, 1)
            incProgress(0.5)

            if (is.null(api_result)) {
              add_log("Extraction échouée.", "ERROR")
            } else {
              add_log(paste0("Réponse reçue en ", elapsed, "s (", nchar(api_result), " car.)"), "OK")
            }
          }
        }
      )

      if (is.null(api_result) || length(img_list) == 0) {
        add_log("Aucun résultat.", "ERROR")
        showNotification("L'extraction a échoué.", type = "error", duration = 8)
      } else {
        md <- build_teacher_markdown(
          teacher, discipline, type_epreuve, section, etablissement, api_result
        )
        rv$current_md      <- md
        rv$current_teacher <- teacher
        rv$all_results[[teacher]] <- list(
          teacher_name  = teacher,
          discipline    = discipline,
          type_epreuve  = type_epreuve,
          section       = section,
          etablissement = etablissement,
          raw_content   = api_result,   # contenu brut extrait (pour le rapport groupé)
          full_md       = md            # markdown complet avec en-tête (pour téléchargement individuel)
        )

        add_log(
          paste0("=== Terminé : ", teacher, " — ", length(img_list), "/", n_pages, " page(s) traitée(s) ==="),
          "OK"
        )

        showNotification(
          paste0(length(img_list), " page(s) traitée(s) avec succès."),
          type = "message", duration = 4
        )

        nav_select("main_tabs", selected = "tab_results", session = session)
      }

    }, error = function(e) {
      add_log(paste("Erreur fatale :", e$message), "ERROR")
      showNotification(paste("Erreur :", e$message), type = "error", duration = 10)
    })

    rv$processing <- FALSE
    enable("process_btn")
  })

  # ── Indicateur de résultats disponibles ───────────────────────────────────────
  output$has_results <- reactive({
    !is.null(rv$current_md) || length(rv$all_results) > 0
  })
  outputOptions(output, "has_results", suspendWhenHidden = FALSE)

  # ── Onglet Résultats ──────────────────────────────────────────────────────────
  output$results_ui <- renderUI({
    if (is.null(rv$current_md)) {
      div(
        class = "d-flex flex-column align-items-center justify-content-center text-muted",
        style = "min-height: 400px; gap: 16px;",
        tags$span(bs_icon("file-earmark-arrow-up", size = "3em")),
        h5("Chargez un PDF et lancez l'extraction"),
        p(
          class = "text-center",
          style = "max-width: 380px;",
          "Configurez votre clé API, renseignez le nom de l'enseignant,",
          " puis sélectionnez le document PDF à analyser."
        )
      )
    } else {
      md_html <- markdown::markdownToHTML(
        text          = rv$current_md,
        fragment.only = TRUE,
        options       = c("use_xhtml", "smartypants", "tables")
      )

      # CSS injecté une seule fois dans le rendu
      result_css <- "
        .extraction-body h1 { font-size:1.4rem; font-weight:700; border-bottom:2px solid #0d6efd; padding-bottom:.4rem; margin-top:0; color:#0d6efd; }
        .extraction-body h2 { font-size:1.1rem; font-weight:600; margin-top:1.6rem; margin-bottom:.6rem; color:#343a40; border-left:4px solid #0d6efd; padding-left:.6rem; }
        .extraction-body h3 { font-size:1rem; font-weight:600; margin-top:1rem; color:#495057; }
        .extraction-body table { width:100%; border-collapse:collapse; font-size:.88rem; margin:.6rem 0 1.2rem; }
        .extraction-body thead th { background:#e9f0ff; color:#212529; font-weight:600; padding:.55rem .75rem; border:1px solid #c7d4f0; text-align:left; }
        .extraction-body tbody tr:nth-child(even) { background:#f8f9ff; }
        .extraction-body tbody tr:hover { background:#eef2ff; }
        .extraction-body td { padding:.45rem .75rem; border:1px solid #dee2e6; vertical-align:top; line-height:1.5; }
        .extraction-body td:nth-child(2) { text-align:center; font-weight:600; white-space:nowrap; }
        .extraction-body td:nth-child(2):contains('Oui') { color:#198754; }
        .extraction-body td:nth-child(3) { white-space:nowrap; color:#6c757d; font-size:.82rem; }
        .extraction-body p { line-height:1.7; margin-bottom:.6rem; }
        .extraction-body em { color:#6c757d; }
        .extraction-body hr { border:none; border-top:1px solid #dee2e6; margin:1.2rem 0; }
      "

      tagList(
        tags$style(result_css),
        div(
          class = "d-flex justify-content-between align-items-center mb-3",
          tags$span(
            class = "fw-bold",
            bs_icon("check-circle-fill", class = "text-success"),
            " ", rv$current_teacher
          ),
          div(
            class = "d-flex gap-2",
            actionButton(
              "edit_current_btn",
              tags$span(bs_icon("pencil-square"), " Modifier"),
              class = "btn-sm btn-outline-primary"
            ),
            actionButton(
              "clear_btn",
              tags$span(bs_icon("x-circle"), " Effacer"),
              class = "btn-sm btn-outline-secondary"
            )
          )
        ),
        div(
          class = "extraction-body",
          style = paste(
            "background:#fff; border:1px solid #dee2e6; border-radius:8px;",
            "padding:28px 32px; overflow-y:auto; max-height:72vh;"
          ),
          HTML(md_html)
        )
      )
    }
  })

  observeEvent(input$clear_btn, {
    rv$current_md      <- NULL
    rv$current_teacher <- NULL
  })

  # ── Éditeur de contenu ────────────────────────────────────────────────────────
  # Ouvre le modal depuis l'onglet Résultats (enseignant actif)
  observeEvent(input$edit_current_btn, {
    req(rv$current_teacher, rv$all_results[[rv$current_teacher]])
    rv$edit_teacher <- rv$current_teacher
    showModal(edit_modal(rv$current_teacher,
                         rv$all_results[[rv$current_teacher]]$raw_content %||% ""))
  })

  # Observe les boutons "Modifier" et "Charger" de l'onglet Documents déjà traités
  observe({
    lapply(names(rv$all_results), function(t) {
      t_id <- gsub("[^[:alnum:]]", "_", t)

      observeEvent(input[[paste0("edit_session_", t_id)]], {
        rv$edit_teacher <- t
        showModal(edit_modal(t, rv$all_results[[t]]$raw_content %||% ""))
      }, ignoreInit = TRUE)

      observeEvent(input[[paste0("load_session_result_", t_id)]], {
        entry <- rv$all_results[[t]]
        rv$current_teacher <- t
        rv$current_md      <- entry$full_md
        nav_select("main_tabs", selected = "tab_results", session = session)
      }, ignoreInit = TRUE)

      observeEvent(input[[paste0("delete_session_", t_id)]], {
        rv$all_results[[t]] <- NULL
        if (!is.null(rv$current_teacher) && rv$current_teacher == t) {
          rv$current_teacher <- NULL
          rv$current_md      <- NULL
        }
        showNotification(paste0(t, " supprimé de la session."),
                         type = "warning", duration = 4)
      }, ignoreInit = TRUE)
    })
  })

  # Sauvegarde les corrections
  observeEvent(input$edit_save, {
    t <- rv$edit_teacher
    req(nzchar(t %||% ""), t %in% names(rv$all_results))

    new_content <- input$edit_textarea
    entry       <- rv$all_results[[t]]

    # Mettre à jour raw_content et reconstruire full_md
    entry$raw_content <- new_content
    entry$full_md     <- build_teacher_markdown(
      entry$teacher_name, entry$discipline, entry$type_epreuve,
      entry$section, entry$etablissement, new_content
    )
    rv$all_results[[t]] <- entry

    # Mettre à jour la vue active si c'est le même enseignant
    if (!is.null(rv$current_teacher) && rv$current_teacher == t) {
      rv$current_md <- entry$full_md
    }

    removeModal()
    rv$edit_teacher <- NULL
    showNotification(paste0("\u2713 Corrections enregistr\u00e9es pour ", t),
                     type = "message", duration = 4)
  })

  observeEvent(input$edit_cancel, {
    removeModal()
    rv$edit_teacher <- NULL
  })

  # Onglet Session ────────────────────────────────────────────────────────────
  output$session_ui <- renderUI({
    teachers <- names(rv$all_results)

    if (length(teachers) == 0) {
      div(class = "text-center text-muted mt-5",
          p("Aucun document traité dans cette session."))
    } else {
      tagList(
        p(class = "text-muted mb-2",
          paste0(length(teachers), " enseignant(s) traité(s) dans cette session :")),
        tags$ul(
          class = "list-group",
          lapply(teachers, function(t) {
            entry   <- rv$all_results[[t]]
            meta    <- paste(Filter(nzchar, c(
              entry$discipline, entry$type_epreuve, entry$section
            )), collapse = " · ")
            n_chars <- nchar(entry$raw_content %||% "")
            t_id    <- gsub("[^[:alnum:]]", "_", t)
            tags$li(
              class = "list-group-item d-flex justify-content-between align-items-center",
              div(
                tags$span(bs_icon("person-check", class = "text-success me-2"), t),
                if (nzchar(meta)) tags$small(class = "text-muted d-block ms-4", meta)
              ),
              div(
                class = "d-flex align-items-center gap-2",
                actionButton(
                  paste0("load_session_result_", t_id),
                  tags$span(bs_icon("box-arrow-up-right")),
                  class = "btn-sm btn-outline-success",
                  title = paste0("Afficher dans Résultats — ", t)
                ),
                actionButton(
                  paste0("edit_session_", t_id),
                  tags$span(bs_icon("pencil-square")),
                  class = "btn-sm btn-outline-primary",
                  title = paste0("Modifier — ", t)
                ),
                actionButton(
                  paste0("delete_session_", t_id),
                  tags$span(bs_icon("trash3")),
                  class = "btn-sm btn-outline-danger",
                  title = paste0("Supprimer — ", t)
                ),
                tags$span(class = "badge bg-success rounded-pill",
                          paste0(format(n_chars, big.mark = " "), " car."))
              )
            )
          })
        )
      )
    }
  })

  # ── Onglet Rapport ─────────────────────────────────────────────────────────

  # Disciplines disponibles (réactif)
  available_disciplines <- reactive({
    if (length(rv$all_results) == 0) return(character(0))
    discs <- vapply(rv$all_results, function(e) {
      d <- e$discipline %||% ""
      as.character(d)[1L]          # toujours un scalaire
    }, character(1))
    sort(unique(Filter(nzchar, discs)))
  })

  output$rapport_ui <- renderUI({
    if (length(rv$all_results) == 0) {
      div(
        class = "d-flex flex-column align-items-center justify-content-center text-muted",
        style = "min-height: 400px; gap: 16px;",
        tags$span(bs_icon("file-earmark-richtext", size = "3em")),
        h5("Traitez au moins un PDF pour générer les rapports"),
        p(class = "text-center", style = "max-width: 420px;",
          "Un rapport PDF séparé sera généré par discipline,",
          " regroupant tous les enseignants par type d'évaluation et section.")
      )
    } else {
      discs <- available_disciplines()
      entries <- rv$all_results

      # Carte par discipline
      disc_cards <- lapply(discs, function(disc) {
        slug  <- gsub("[^[:alnum:]_]", "_", disc)
        items <- Filter(function(e) identical(e$discipline %||% "", disc), entries)

        # Groupes dans cette discipline
        grp_tbl <- table(sapply(items, function(e)
          paste(Filter(nzchar, c(e$type_epreuve, e$section)), collapse = " · ")
        ))

        div(
          class = "card mb-3",
          div(
            class = "card-header d-flex justify-content-between align-items-center",
            style = "background:#1a3a5c; color:white;",
            tags$span(
              tags$strong(bs_icon("book"), " ", disc),
              tags$span(
                class = "ms-3 badge",
                style = "background:rgba(255,255,255,.2);",
                paste0(length(items), " enseignant", if(length(items)>1) "s" else "")
              )
            ),
            downloadButton(
              paste0("dl_pdf_", slug),
              tags$span(bs_icon("file-earmark-pdf"), " Télécharger PDF"),
              class = "btn-sm btn-light fw-bold"
            )
          ),
          div(
            class = "card-body py-2",
            tags$ul(
              class = "list-group list-group-flush",
              lapply(names(sort(grp_tbl)), function(g) {
                n <- grp_tbl[[g]]
                tags$li(
                  class = "list-group-item d-flex justify-content-between py-1",
                  style = "font-size:.9rem;",
                  tags$span(bs_icon("folder2-open", class = "text-primary me-2"), g),
                  tags$span(class = "badge bg-secondary rounded-pill",
                            paste0(n, " ens."))
                )
              })
            )
          )
        )
      })

      tagList(
        div(class = "row g-3 mb-4",
          div(class = "col-auto",
            div(class = "card border-primary text-center px-4 py-2",
              h3(class = "mb-0 text-primary", length(rv$all_results)),
              tags$small(class = "text-muted", "enseignants"))),
          div(class = "col-auto",
            div(class = "card border-0 bg-light text-center px-4 py-2",
              h3(class = "mb-0", length(discs)),
              tags$small(class = "text-muted", "disciplines")))
        ),
        div(disc_cards)
      )
    }
  })

  # Handlers PDF dynamiques — un par discipline
  # Enregistrés dès qu'une nouvelle discipline apparaît dans all_results
  registered_pdf_handlers <- character(0)

  observe({
    discs <- available_disciplines()
    new_discs <- setdiff(discs, registered_pdf_handlers)
    if (length(new_discs) == 0) return()

    lapply(new_discs, function(disc) {
      local({
        d    <- disc
        slug <- gsub("[^[:alnum:]_]", "_", d)

        output[[paste0("dl_pdf_", slug)]] <- downloadHandler(
          filename = function() {
            paste0("rapport_", slug, "_", format(Sys.time(), "%Y%m%d"), ".pdf")
          },
          content = function(file) {
            filtered <- Filter(function(e) identical(e$discipline %||% "", d), rv$all_results)
            n_ens <- length(filtered)
            add_log(paste0("Génération PDF — ", d, " (", n_ens, " enseignant(s))…"))
            showNotification(
              paste0("Génération du PDF \u00ab\u00a0", d, "\u00a0\u00bb\u2026"),
              id = "pdf_notif", duration = NULL, type = "message"
            )

            rmd_content <- generate_rmd_report(
              filtered,
              title = paste0("Analyse qualitative des pr\u00e9tests \u2014 ", d)
            )
            tmp_rmd <- tempfile(fileext = ".Rmd")
            tmp_pdf <- tempfile(fileext = ".pdf")
            on.exit({
              for (f in c(tmp_rmd, tmp_pdf)) {
                if (file.exists(f)) try(file.remove(f), silent = TRUE)
              }
              removeNotification("pdf_notif")
            }, add = TRUE)
            writeLines(rmd_content, tmp_rmd, useBytes = TRUE)

            tryCatch({
              rmarkdown::render(
                tmp_rmd,
                output_file = tmp_pdf,
                quiet       = TRUE,
                envir       = new.env(parent = globalenv())
              )
              file.copy(tmp_pdf, file, overwrite = TRUE)
              add_log(paste0("PDF pr\u00eat : ", d), "OK")
            }, error = function(e) {
              add_log(paste0("Erreur PDF (", d, ") : ", e$message), "ERROR")
              stop(e)
            })
          }
        )
      })
    })

    registered_pdf_handlers <<- c(registered_pdf_handlers, new_discs)
  })

  # ── Journal ───────────────────────────────────────────────────────────────────
  output$log_output <- renderUI({
    lines <- if (length(rv$log) == 0) "En attente…" else paste(rv$log, collapse = "\n")
    tags$pre(
      style = paste(
        "background:#1e1e1e; color:#d4d4d4;",
        "border:none; border-radius:8px; padding:16px;",
        "font-family:monospace; font-size:.82rem; line-height:1.5;",
        "max-height:70vh; overflow-y:auto; white-space:pre-wrap;"
      ),
      lines
    )
  })

  # ── Liste des sauvegardes disponibles (réactif) ───────────────────────────
  available_saves <- reactive({
    input$save_session  # se rafraîchit après chaque sauvegarde
    if (!dir.exists(SAVE_DIR)) return(character(0))
    sort(list.files(SAVE_DIR, pattern = "\\.rds$"), decreasing = TRUE)
  })

  output$load_session_ui <- renderUI({
    saves <- available_saves()
    if (length(saves) == 0) {
      tags$p(class = "text-muted small mb-2", "Aucune sauvegarde disponible.")
    } else {
      selectInput(
        "load_session_select",
        label    = NULL,
        choices  = saves,
        selected = saves[1]
      )
    }
  })

  # ── Sauvegarde de session (.rds) ──────────────────────────────────────────
  observeEvent(input$save_session, {
    req(length(rv$all_results) > 0)
    dir.create(SAVE_DIR, recursive = TRUE, showWarnings = FALSE)
    fname <- paste0("session_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".rds")
    fpath <- file.path(SAVE_DIR, fname)
    tryCatch({
      saveRDS(rv$all_results, fpath)
      add_log(paste0("Session sauvegard\u00e9e : ", fname,
                     " (", length(rv$all_results), " enseignant(s))"), "OK")
      showNotification(
        paste0("\u2713 Sauvegard\u00e9 : ", fname),
        type = "message", duration = 5
      )
    }, error = function(e) {
      showNotification(paste("Erreur sauvegarde :", e$message), type = "error", duration = 8)
    })
  })

  # ── Chargement de session (.rds) ──────────────────────────────────────────
  observeEvent(input$load_session_btn, {
    req(input$load_session_select)
    path <- file.path(SAVE_DIR, input$load_session_select)
    if (!file.exists(path)) {
      showNotification("Fichier introuvable.", type = "error", duration = 6)
      return()
    }

    loaded <- tryCatch(
      readRDS(path),
      error = function(e) {
        showNotification(paste("Fichier invalide :", e$message), type = "error", duration = 8)
        NULL
      }
    )
    if (is.null(loaded)) return()

    if (!is.list(loaded) || length(loaded) == 0 ||
        !all(sapply(loaded, function(e) is.list(e) && "teacher_name" %in% names(e)))) {
      showNotification(
        "Le fichier ne semble pas \u00eatre une session valide.",
        type = "error", duration = 6
      )
      return()
    }

    n_new      <- length(setdiff(names(loaded), names(rv$all_results)))
    n_replaced <- length(intersect(names(loaded), names(rv$all_results)))

    for (key in names(loaded)) {
      rv$all_results[[key]] <- loaded[[key]]
    }

    msg <- paste0(
      "Session charg\u00e9e : ", length(loaded), " enseignant(s)",
      if (n_new > 0)      paste0(" (", n_new, " nouveaux"),
      if (n_replaced > 0) paste0(", ", n_replaced, " mis \u00e0 jour"),
      if (n_new > 0 || n_replaced > 0) ")" else ""
    )
    add_log(msg, "OK")
    showNotification(msg, type = "message", duration = 5)
    nav_select("main_tabs", selected = "tab_session", session = session)
  })
}

# ── Lancement ──────────────────────────────────────────────────────────────────
shinyApp(ui, server)
