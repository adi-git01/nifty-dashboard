# What NSE gives for free -- probe of 20 Nov 2026 from GitHub Actions

Run at 21 Nov 2026 21:52 IST. ✅ = reachable from a GitHub runner; ❌ = refused or missing.

## Daily archive files (no login)

| Source | Status | Rows | Columns | Use |
|---|---|---|---|---|
| CM bhavcopy (UDiFF) | ✅ | 3,742 | TradDt, BizDt, Sgmt, Src, FinInstrmTp, FinInstrmId, ISIN, TckrSymb, SctySrs, XpryDt, FininstrmActlXpryDt, StrkPric, OptnTp, FinInstrmNm … | official OHLC, volume, turnover, trades for every security |
| Security-wise delivery (sec_bhavdata_full) | ✅ | 3,583 | SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, LAST_PRICE, CLOSE_PRICE, AVG_PRICE, TTL_TRD_QNTY, TURNOVER_LACS, NO_OF_TRADES, DELIV_QTY … | OHLC + DELIVERY quantity and %: real buying vs intraday churn |
| MTO delivery file | ⚠️ got 131,561 bytes, unreadable (ParserError) | | | delivery position (older format) |
| 52-week high/low | ⚠️ got 325,335 bytes, unreadable (ParserError) | | | official 52W high/low per stock |
| Index closes (all indices) | ✅ | 169 | Index Name, Index Date, Open Index Value, High Index Value, Low Index Value, Closing Index Value, Points Change, Change(%), Volume, Turnover (Rs. Cr.), P/E, P/B, Div Yield | close, P/E, P/B, dividend yield of every NSE index (sector indices too) |
| Bulk deals (today) | ✅ | 108 | Date, Symbol, Security Name, Client Name, Buy/Sell, Quantity Traded, Trade Price / Wght. Avg. Price, Remarks | large disclosed deals: who bought/sold |
| Block deals (today) | ✅ | 5 | Date, Symbol, Security Name, Client Name, Buy/Sell, Quantity Traded, Trade Price / Wght. Avg. Price | block-window deals |
| Short selling | ✅ | 84 | Security Name, Symbol Name, Trade Date, Quantity | shares sold short per stock |
| Price band changes | ✅ | 7 | Sr. No, Symbol, Series, Security Name, From, To | circuit band changes |
| Daily volatility (CMVOLT) | ✅ | 5,002 | Date, Symbol, Underlying Close Price (A), Underlying Previous Day Close Price (B), Underlying Log Returns (C) = LN(A/B), Previous Day Underlying Volatility (D), Current Day Underlying Daily Volatility (E) = Sqrt(0.995*D*D + 0.005*C*C), Underlying Annualised Volatility (F) = E*Sqrt(365) | daily and annualised volatility |
| F&O ban list | ❌ HTTP 404 | | | stocks in the F&O ban period |
| F&O participant OI | ✅ | 6 | Participant wise Open Interest (no. of contracts) in Equity Derivatives as on Nov 20, 2026"", Unnamed: 2, Unnamed: 3, Unnamed: 4, Unnamed: 5, Unnamed: 6, Unnamed: 7, Unnamed: 8, Unnamed: 9, Unnamed: 10, Unnamed: 11, Unnamed: 12, Unnamed: 13 … | FII / DII / Pro / Client positions in index & stock futures/options |
| F&O bhavcopy (UDiFF) | ✅ | 45,082 | TradDt, BizDt, Sgmt, Src, FinInstrmTp, FinInstrmId, ISIN, TckrSymb, SctySrs, XpryDt, FininstrmActlXpryDt, StrkPric, OptnTp, FinInstrmNm … | futures/options OHLC and open interest |

## JSON endpoints behind nseindia.com (session cookie, no account)

| Source | Status | Records | Fields | Use |
|---|---|---|---|---|
| FII / DII cash activity | ✅ | 2 | buyValue, category, date, netValue, sellValue | net buying by FIIs and DIIs in the cash market |
| Corporate actions (upcoming) | ✅ | 7 | bcEndDate, bcStartDate, caBroadcastDate, comp, exDate, faceVal, ind, isin, ndEndDate, ndStartDate, recDate, series | splits, bonuses, dividends with ex-dates |
| Board meetings / results calendar | ✅ | 20 | bm_symbol, bm_date, bm_purpose, bm_desc, sm_indusrty, bm_timestamp, sm_name, sm_isin, diff, sysTime, attFileSize, attachment | results dates ahead |
| Event calendar | ✅ | 21 | symbol, company, purpose, bm_desc, date | results, AGMs, dividends |
| Insider trading (PIT) | ✅ | 0 |  | promoter / insider buys and sells |
| Pledged shares | ✅ | 0 |  | promoter pledge levels |
| ASM list (surveillance) | ✅ | 1 | longterm, shortterm | stocks under additional surveillance |
| GSM list (surveillance) | ✅ | 76 | companyName, gsmStage, gsmTime, isin, survCode, survDesc, symbol, srno | graded surveillance |
| Announcements | ✅ | 20 | an_dt, attFilePath, attFileSize, attchmntFile, attchmntText, bflag, csvName, desc, difference, dt, exchdisstime, fileSize | filings: orders, results, management changes |
| Market status | ✅ | 6 | market, marketStatus, tradeDate, index, last, variation, percentChange, marketStatusMessage | is the market open, last session |
| Quote: RELIANCE | ❌ HTTP 403 | | | P/E, sector P/E, shares issued, industry classification |
| Results comparison: RELIANCE | ✅ | 5 | re_amt_grs_np_asst, re_bal_rbi_oth_bnk_funds, re_basic_eps, re_basic_eps_for_cont_dic_opr, re_bsc_eps_bfr_exi, re_cap_ade_rat, re_cet_1_ret, re_con_pro_loss, re_create_dt, re_curr_tax, re_debt_eqt_rat, re_debt_rdmption | last quarters' P&L lines: revenue, costs, profit, EPS |
| Financial results list | ✅ | 3,816 | audited, bank, broadCastDate, companyName, consolidated, cumulative, difference, exchdisstime, filingDate, financialYear, format, fromDate | latest quarterly results filed |
| Shareholding: RELIANCE | ✅ | 23 | broadcastDate, cgTimeStamp, countString, date, desc, employeeTrusts, format, index, industry, isin, name, newColumn | promoter / public holding by quarter |
| Nifty 50 constituents live | ❌ HTTP 404 | | | live prices of an index's stocks |

## Saved: `data/nse/eod_20261120.csv.gz` -- 3,461 stocks (EQ/BE/BZ/SME series)

Delivery %: median 60% · stocks with ≥ 60% delivery: 1,625

## Yahoo snapshot vs NSE official close (20 Nov)

- 1,141 of 1,168 snapshot stocks matched; median gap 0.000%
- within 0.1%: 100% · off by more than 0.5%: 0 · not in NSE's EQ list: 27

## Samples

**CM bhavcopy (UDiFF)**

```
       TradDt       BizDt Sgmt  Src FinInstrmTp  FinInstrmId          ISIN  ... SsnId NewBrdLotQty  Rmks  Rsvd1  Rsvd2  Rsvd3 Rsvd4
0  2026-11-20  2026-11-20   CM  NSE         STK        19078  IN0020200104  ...    F1            1   NaN    NaN    NaN    NaN   NaN
1  2026-11-20  2026-11-20   CM  NSE         STK         1342  IN0020200286  ...    F1            1   NaN    NaN    NaN    NaN   NaN
2  2026-11-20  2026-11-20   CM  NSE         STK         3742  IN0020210053  ...    F1            1   NaN    NaN    NaN    NaN   NaN
```

**Security-wise delivery (sec_bhavdata_full)**

```
       SYMBOL SERIES         DATE1  PREV_CLOSE  OPEN_PRICE  HIGH_PRICE  LOW_PRICE  ...  CLOSE_PRICE  AVG_PRICE  TTL_TRD_QNTY  TURNOVER_LACS  NO_OF_TRADES  DELIV_QTY DELIV_PER
0   20MICRONS     EQ   20-Nov-2026      180.30       180.0      181.77     177.72  ...       179.49     179.55         34452          61.86          1470      17830     51.75
1  21STCENMGM     BE   20-Nov-2026       48.51        47.0       47.00      46.09  ...        46.09      46.20          6249           2.89            62          -         -
2      360ONE     EQ   20-Nov-2026     1197.90      1194.0     1224.60    1191.55  ...      1208.10    1212.82       1133648       13749.16         32363     545332     48.10
```

**Index closes (all indices)**

```
      Index Name  Index Date Open Index Value High Index Value Low Index Value  Closing Index Value  Points Change Change(%)      Volume Turnover (Rs. Cr.)    P/E   P/B Div Yield
0       Nifty 50  20-11-2026            25368          25808.8         25345.2             25800.60         286.30      1.12   428344270           42565.93  21.02  3.17      1.32
1  Nifty Next 50  20-11-2026          72338.8         73227.85         72129.2             73207.25         317.50       .44   622436342           13756.28  17.75  3.52      1.37
2      Nifty 100  20-11-2026         26238.95         26664.65        26205.45             26662.85         263.95         1  1050780612           56322.21  20.35  3.22      1.33
```

**Bulk deals (today)**

```
          Date   Symbol         Security Name              Client Name Buy/Sell  Quantity Traded  Trade Price / Wght. Avg. Price Remarks
0  20-NOV-2026  BLSPOLY  BLS Polymers Limited  ALPHAGREP SECURITIES...      BUY           414422                   273.30              -
1  20-NOV-2026  BLSPOLY  BLS Polymers Limited        QE SECURITIES LLP      BUY          1071662                   272.55              -
2  20-NOV-2026  BLSPOLY  BLS Polymers Limited  JUMP TRADING FINANCI...      BUY           612007                   272.23              -
```

**Block deals (today)**

```
          Date    Symbol           Security Name              Client Name Buy/Sell  Quantity Traded  Trade Price / Wght. Avg. Price
0  20-NOV-2026  GALLANTT  Gallantt Ispat Limited  SHREE SURABHI FLOUR ...      BUY           550000                   523.00       
1  20-NOV-2026  GALLANTT  Gallantt Ispat Limited  SURYALAXMI INFRACON ...      BUY          1060292                   524.41       
2  20-NOV-2026     INDGN        Indegene Limited    RAJESH BHASKARAN NAIR      BUY           380750                   657.00       
```

**Short selling**

```
             Security Name Symbol Name   Trade Date  Quantity
0      360 ONE WAM LIMITED      360ONE  19-Nov-2026         7
1  BILLIONBRAINS GARAGE...       GROWW  19-Nov-2026        12
2         LT FOODS LIMITED     LTFOODS  19-Nov-2026        50
```

**Price band changes**

```
   Sr. No      Symbol Series            Security Name  From  To
0       1  SHRINGARMS     EQ  SHRINGAR HOUSE OF MA...    10   5
1       2     TECHERA     SM  TECHERA ENGINEERING ...    10   5
2       3      BALAXI     EQ  BALAXI PHARMACEUTICA...    20  10
```

**Daily volatility (CMVOLT)**

```
          Date      Symbol Underlying Close Price (A) Underlying Previous Day Close Price (B) Underlying Log Returns (C) = LN(A/B) Previous Day Underlying Volatility (D) Current Day Underlying Daily Volatility (E) = Sqrt(0.995*D*D + 0.005*C*C) Underlying Annualised Volatility (F) = E*Sqrt(365)
0  20-NOV-2026   20MICRONS                   179.48                     180.31                                 -0.0046                               0.0299                                 0.0298                                                                    0.5693                          
1  20-NOV-2026  21STCENMGM                    46.09                      48.51                                 -0.0512                               0.0238                                 0.0241                                                                    0.4604                          
2  20-NOV-2026      360ONE                  1208.10                    1197.90                                  0.0085                               0.0202                                 0.0202                                                                    0.3859                          
```

**F&O participant OI**

```
  Participant wise Open Interest (no. of contracts) in Equity Derivatives as on Nov 20             2026""          Unnamed: 2         Unnamed: 3               Unnamed: 4              Unnamed: 5             Unnamed: 6  ...              Unnamed: 9            Unnamed: 10              Unnamed: 11             Unnamed: 12              Unnamed: 13            Unnamed: 14 Unnamed: 15
0              Client Type                                                              Future Index Long  Future Index Short  Future Stock Long  Future Stock Short  ...  Option Index Call Long  Option Index Put Long  ...  Option Stock Call Long  Option Stock Put Long  Option Stock Call Short  Option Stock Put Short  Total Long Contracts...  Total Short Contracts         NaN
1                   Client                                                                         155681              114419            3127092                   371549                 3072591                4103061  ...                 1984112                1146937                  1199379                 1194530                 13589474               10578356         NaN
2                      DII                                                                          52405               35254             460150                  4352471                    6837                  99777  ...                    5922                  43360                   483207                   29494                   668451                4904416         NaN
```

**F&O bhavcopy (UDiFF)**

```
       TradDt       BizDt Sgmt  Src FinInstrmTp  FinInstrmId  ISIN  ... SsnId  NewBrdLotQty Rmks Rsvd1  Rsvd2 Rsvd3 Rsvd4
0  2026-11-20  2026-11-20   FO  NSE         STO        68455   NaN  ...    F1          1550  NaN   NaN    NaN   NaN   NaN
1  2026-11-20  2026-11-20   FO  NSE         STO        70776   NaN  ...    F1          1550  NaN   NaN    NaN   NaN   NaN
2  2026-11-20  2026-11-20   FO  NSE         STO        68374   NaN  ...    F1          1550  NaN   NaN    NaN   NaN   NaN
```

**FII / DII cash activity**

```
   buyValue category         date  netValue sellValue
0  18492.63      DII  20-Nov-2026   -4859.8  23352.43
1   23596.4  FII/FPI  20-Nov-2026  10308.42  13287.98
```

**Corporate actions (upcoming)**

```
  bcEndDate bcStartDate caBroadcastDate                     comp       exDate faceVal ind          isin ndEndDate ndStartDate      recDate series                  subject     symbol
0         -           -            None      GOVERNMENT OF INDIA  23-Nov-2026     100   -  IN0020140060         -           -  23-Nov-2026     GS         Interest Payment  815GS2026
1         -           -            None  Pashupati Cotspin Li...  23-Nov-2026       1   -  INE124Y01010         -           -  23-Nov-2026     EQ  Interim Dividend - R...  PASHUPATI
2         -           -            None  Zee Entertainment En...  25-Nov-2026       1   -  INE256A01028         -           -  25-Nov-2026     EQ                 Buy Back       ZEEL
```

**Board meetings / results calendar**

```
   bm_symbol      bm_date               bm_purpose                  bm_desc sm_indusrty          bm_timestamp                  sm_name  ...               attachment intimationType                    ixbrl ixbrlFileSize meetingType oriiginalMeetingDate proposedMeetingDate
0  DAVANGERE  26-Nov-2026  Board Meeting Intima...  Davangere Sugar Comp...           -  21-Nov-2026 18:29:19  Davangere Sugar Comp...  ...  https://nsearchives....           None  https://nsearchives....       3.12 KB        None                 None                None
1  DAVANGERE  26-Nov-2026   Other business matters  To consider and disc...           -  21-Nov-2026 18:22:55  Davangere Sugar Comp...  ...  https://nsearchives....           None                     None          None        None                 None                None
2   RAJRATAN  25-Nov-2026   Other business matters  To consider other bu...           -  21-Nov-2026 17:17:03  Rajratan Global Wire...  ...  https://nsearchives....           None                     None          None        None                 None                None
```

**Event calendar**

```
      symbol                  company                  purpose                  bm_desc         date
0      BCSSL  Bombay Coated and Sp...        Financial Results  To consider and appr...  23-Nov-2026
1  DALBHARAT    Dalmia Bharat Limited  Buyback/Other busine...  To consider buyback ...  23-Nov-2026
2    SFCTECH  SFC Environmental Te...        Financial Results  To consider and appr...  23-Nov-2026
```

**ASM list (surveillance)**

```
                  longterm                shortterm
0  {'data': [{'asmSurvI...  {'data': [{'asmSurvI...
```

**GSM list (surveillance)**

```
               companyName gsmStage               gsmTime          isin                 survCode                 survDesc      symbol  srno
0  AGS Transact Technol...     LXII  20-Nov-2026 08:10:03  INE583L01014  IBC - Receipt & GSM ...  Insolvency and Bankr...      AGSTRA     1
1  Ankit Metal & Power ...    LVIII  20-Nov-2026 08:10:03  INE106I01010       IBC I & GSM 0 (58)  ASM IBC Stage I and ...  ANKITMETAL     2
2     Aqylon Nexus Limited        L  20-Nov-2026 08:10:03  INE416A01051     LTASM I & GSM 0 (50)  LTASM Stage I and GS...      AQYLON     3
```

**Announcements**

```
                  an_dt attFilePath attFileSize             attchmntFile             attchmntText bflag csvName  ... smIndustry       sm_isin                  sm_name            sort_date     symbol             voiceSummary                 voiceUrl
0  21-Nov-2026 21:50:17        None   503.07 KB  https://nsearchives....  ESDS Software Soluti...  None    None  ...       None  INE0DRI01029  ESDS Software Soluti...  2026-11-21 21:50:17       ESDS  ESDS Software Soluti...  https://nsearchives....
1  21-Nov-2026 21:26:32        None   326.91 KB  https://nsearchives....  Intellect Design Are...  None    None  ...       None  INE306R01017  Intellect Design Are...  2026-11-21 21:26:32  INTELLECT  Intellect Design Are...  https://nsearchives....
2  21-Nov-2026 21:23:04        None   205.74 KB  https://nsearchives....  Coforge Limited has ...  None    None  ...       None  INE591G01017          Coforge Limited  2026-11-21 21:23:04    COFORGE  Coforge Limited has ...  https://nsearchives....
```

**Market status**

```
           market marketStatus          tradeDate     index     last variation percentChange marketStatusMessage
0  Capital Market        Close  20-Nov-2026 15:30  NIFTY 50  25800.6     286.3          1.12    Market is Closed
1        Currency        Close        20-Nov-2026                                               Market is Closed
2       Commodity        Close        20-Nov-2026                                               Market is Closed
```

**Results comparison: RELIANCE**

```
  re_amt_grs_np_asst re_bal_rbi_oth_bnk_funds re_basic_eps re_basic_eps_for_cont_dic_opr re_bsc_eps_bfr_exi re_cap_ade_rat re_cet_1_ret  ... re_staff_cost  re_tax re_tax_expens_of_dis_opr     re_to_dt re_tot_exp_exc_pro_cont re_tot_inc re_total_inc
0               None                     None         None                     6.44                    None           None         None  ...        218100  287600                        0  31-DEC-2024                    None       None     13147400
1               None                     None         None                     11.4                    None           None         None  ...        207700  250100                        0  30-SEP-2024                    None       None     13785500
2               None                     None         None                    11.25                    None           None         None  ...        215600  253500                        0  30-JUN-2024                    None       None     13783300
```

**Financial results list**

```
      audited bank         broadCastDate              companyName      consolidated      cumulative difference  ...     relatingTo resultDescription resultDetailedDataLink seqNumber      symbol       toDate                     xbrl
0  Un-Audited    N  30-Jul-2026 17:17:53  V.S.T Tillers Tracto...      Consolidated  Non-cumulative   00:00:49  ...  Third Quarter              None                   None   1197616  VSTTILLERS  31-Dec-2024  https://nsearchives....
1  Un-Audited    N  30-Jul-2026 17:15:56  V.S.T Tillers Tracto...  Non-Consolidated  Non-cumulative   00:00:32  ...  Third Quarter              None                   None   1197614  VSTTILLERS  31-Dec-2024  https://nsearchives....
2  Un-Audited    N  25-Jun-2026 16:39:17  Videocon Industries ...  Non-Consolidated  Non-cumulative   00:00:59  ...  Third Quarter              None                   None   1197606    VIDEOIND  31-Dec-2024  https://nsearchives....
```

**Shareholding: RELIANCE**

```
          broadcastDate cgTimeStamp countString         date   desc employeeTrusts format  ...    symbol            systemDate timeDifference typeOfSubmission  underlyingDrs                     xbrl xbrlFileSize
0  21-OCT-2026 16:25:15        None        None  30-SEP-2026  NEW_1              0   None  ...  RELIANCE  21-OCT-2026 16:25:19       00:00:04             None           None  https://nsearchives....    511.30 KB
1  16-JUL-2026 19:24:44        None        None  30-JUN-2026  NEW_1              0   None  ...  RELIANCE  16-JUL-2026 19:24:48       00:00:04             None           None  https://nsearchives....    510.29 KB
2  21-APR-2026 13:25:14        None        None  31-MAR-2026  NEW_1              0   None  ...  RELIANCE  21-APR-2026 13:25:18       00:00:04             None           None  https://nsearchives....    511.88 KB
```

