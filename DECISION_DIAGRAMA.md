```mermaid
flowchart TD
  A[Inicio: Frame] --> B[Detectar contornos de posible carta]
  B --> C{Obtener perspectiva (warp)}
  C --> D[Extraer esquina superior izquierda (corner)]
  D --> E[Preprocesar corner (grayscale, resize, threshold)]
  E --> F[MatchTemplate para Rank -> best_rank, score_rank]
  E --> G[MatchTemplate para Suit -> best_suit, score_suit]
  F --> H{score_rank >= umbral_rank?}
  G --> I{score_suit >= umbral_suit?}
  H -- Sí --> J[Rank = best_rank]
  H -- No --> K[Rank = "unknown"]
  I -- Sí --> L[Suit = best_suit]
  I -- No --> M[Suit = "unknown"]
  J --> N[Combinar Rank/Suit -> Etiqueta]
  K --> N
  L --> N
  M --> N
  N --> O{Ambos desconocidos?}
  O -- Sí --> P[Fallback: intentar heurísticas (color de carta, OCR, template multi-scale, reintentar)]
  O -- No --> Q[Salida: etiqueta final y confidence report]
  P --> Q
  Q --> R[Actualizar UI / ventanas (last_card, last_rank, last_suit)]
  R --> S[Fin]
```

Explicación corta:
- El diagrama muestra el flujo desde la detección de contornos hasta la decisión final usando los umbrales MATCH_THRESHOLD_RANK y MATCH_THRESHOLD_SUIT.
- Incluye nodos de fallback (reintentos, heurísticas) y el reporte de confianza útil para debug.
