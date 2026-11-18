Prompt version 2026-11-10.haiku.2

You check whether retrieved text states the facts of gold evidence sentences.

Use only the retrieved items. Do not use outside knowledge and do not reason
your way to a fact the text does not state; plain rewording, synonyms,
abbreviations and different formatting are fine.

For every fact give one status:
- stated: some item says it, with every specific matching (names, numbers,
  dates, places, who did what, qualifiers).
- contradicted: some item gives a different value for the same specific, such
  as another year or another person.
- missing: neither of the above.
If one item states the fact and another contradicts it, the status is contradicted.

Evidence for stated or contradicted facts: copy one continuous passage from
one item, character for character, with no ellipses or edits, and give that
item's number. Never join separate words into a quote that does not appear as
written; if the words are apart, quote the longer continuous passage that
contains them. A fact spread across items may take several quotes. The quotes
together must contain the words that state every specific of the fact (who,
what, numbers, dates, reasons); a quote that only sets the topic, or stops
before the specific, is not evidence. Missing facts take no evidence. Every quote is checked against the item text by a
program, and a quote that does not appear there counts as missing.

Write the verdict file named in your task as a JSON array with one object per
item, in any order:
{"id": "<item id>", "facts": [{"fact": "S.F", "status": "stated", "evidence": [{"item": 3, "quote": "..."}]}, ...]}
where "S.F" is the fact's label as given (sentence number, fact number).
Give exactly one entry for every fact label of the item. Use only ids from
this file and do not skip any item.

This file has 6 items.

=== ITEM id=645e50a042847762 ===
Gold sentences and facts:
Sentence 0: Celebrity Notability ( known for ) Professional partner Status Trista Sutter Reality television star Louis Van Amstel Eliminated 1st on June 8 , 2005 Evander Holyfield Professional boxer Edyta Sliwinska Eliminated 2nd on June 15 , 2005 Rachel Hunter Supermodel & actress Jonathan Roberts Eliminated 3rd on June 22 , 2005 Joey McIntyre New Kids on the Block singer Ashly DelGrosso Third place on June 29 , 2005 John O'Hurley Actor & game show host Charlotte Jorgensen Runner - up on July 6 , 2005 Kelly Monaco Actress Alec Mazo Winner on July 6 , 2005
  Fact 0.0: Trista Sutter is known for being a reality television star.
  Fact 0.1: Trista Sutter's professional partner is Louis Van Amstel.
  Fact 0.2: Trista Sutter was eliminated 1st on June 8, 2005.
  Fact 0.3: Evander Holyfield is a professional boxer.
  Fact 0.4: Evander Holyfield's professional partner is Edyta Sliwinska.
  Fact 0.5: Evander Holyfield was eliminated 2nd on June 15, 2005.
  Fact 0.6: Rachel Hunter is a supermodel and actress.
  Fact 0.7: Rachel Hunter's professional partner is Jonathan Roberts.
  Fact 0.8: Rachel Hunter was eliminated 3rd on June 22, 2005.
  Fact 0.9: Joey McIntyre is a New Kids on the Block singer.
  Fact 0.10: Joey McIntyre's professional partner is Ashly DelGrosso.
  Fact 0.11: Joey McIntyre finished in third place on June 29, 2005.
  Fact 0.12: John O'Hurley is an actor and game show host.
  Fact 0.13: John O'Hurley's professional partner is Charlotte Jorgensen.
  Fact 0.14: John O'Hurley finished as runner-up on July 6, 2005.
  Fact 0.15: Kelly Monaco is an actress.
  Fact 0.16: Kelly Monaco's professional partner is Alec Mazo.
  Fact 0.17: Kelly Monaco was the winner on July 6, 2005.

Retrieved items:
<item 1>
Dancing with the Stars (U.S. season 1)
Dancing with the Stars ( U.S. season 1 ) - wikipedia Dancing with the Stars ( U.S. season 1 ) Jump to : navigation , search Dancing with the Stars ( season 1 )
Country of origin United States No. of episodes 8 Release Original network ABC Original release June 1 -- July 6 , 2005 Additional information Celebrity winner
Kelly Monaco Professional winner Alec Mazo Season chronology Next → Season 2 The first season of Dancing with the Stars ( U.S. ) debuted on ABC on June 1 , 2005
. Six celebrities were paired with six professional ballroom dancers . Tom Bergeron and Lisa Canning were the hosts for this season . The judges were Carrie Ann
Inaba , Len Goodman , and Bruno Tonioli . The premiere drew over 13 million viewers , the second biggest summer debut ever for an American reality series after
Survivor . The second week climbed to 15 million viewers , and the show climbed to No. 1 in the summer 2005 TV season ratings , where it remained for the
remainder of the season . The season finale aired July 6 , 2005 .
</item 1>

<item 2>
Dancing with the Stars (U.S. season 1)
Rank by average Place Couple Total Number of dances Average John & Charlotte 199 8 24.9 Rachel & Jonathan 95 23.8 Kelly & Alec 179 8 22.4 Joey & Ashly 128 6
21.3 5 6 Trista & Louis 37 18.5 6 5 Evander & Edyta 45 15.0 Pro 's average chart ( edit ) Rank by average Place Couple Average Charlotte Jorgensen 24.9 6 Louis
van Amstel 21.8 Alec Mazo 20.6 Jonathan Roberts 19.4 5 5 Edyta Sliwinska 19.1 6 Ashly DelGrosso 16.
</item 2>

<item 3>
Dancing with the Stars (U.S. season 1)
They finished third . Scoring chart ( edit ) Couple Place 1 + 2 5 6 Kelly & Alec 13 17 30 21 26 22 + 25 = 47 25 + 30 = 55 John & Charlotte 20 26 46 24 21 27 +
27 = 54 27 + 27 = 54 Joey & Ashly 20 21 41 22 20 20 + 25 = 45 Rachel & Jonathan 20 24 44 26 25 Evander & Edyta 5 18 14 32 13 Trista & Louis 6 18 19 37 indicates
the eliminated couple that week indicates the returning couple that was last to be called safe indicates the winning couple indicates the runner - up couple
indicates the third place couple Red numbers indicate the couples with the lowest score for each week Green numbers indicate the couples with the highest score
for each week Average chart ( edit ) This table only counts for dances scored on a traditional 30 - point scale . Rank by average Place Couple Total Number of
dances Average John & Charlotte 199 8 24.
</item 3>

<item 4>
Dancing with the Stars (U.S. season 1)
6 , 7 ) Waltz `` Three Times A Lady '' -- Lionel Richie Evander & Edyta 18 ( 5 , 7 , 6 ) Cha - cha - cha `` Respect '' -- Aretha Franklin Kelly & Alec 13 ( 5 ,
4 , 4 ) Waltz `` I Have Nothing '' -- Whitney Houston John & Charlotte 20 ( 7 , 7 , 6 ) Cha - cha - cha `` September '' -- Earth , Wind & Fire Trista & Louis 18
( 6 , 6 , 6 ) Waltz `` Come Away with Me '' -- Norah Jones Week 2 ( edit ) Running order Couple Score Dance Music Result Rachel & Jonathan 24 ( 8 , 8 , 8 )
Rumba `` I Do n't Wanna Miss a Thing '' -- Aerosmith from Armageddon Bottom two Joey & Ashly 21 ( 8 , 7 , 6 ) Quickstep `` You 're The One That I Want '' --
John Travolta and Olivia Newton - John Safe Trista & Louis 19 ( 6 , 7 ,
</item 4>

<item 5>
Dancing with the Stars (U.S. season 1)
Jump up ^ `` Dancing with the Stars : Episode 105 '' . TV.com . Retrieved 2009 - 05 - 06 . Jump up ^ `` Dancing with the Stars : Episode 106 '' . TV.com .
Retrieved 2009 - 05 - 06 . External links ( edit ) Official website ( hide ) Dancing with the Stars ( U.S. TV series ) Seasons 5 6 7 8 9 10 11 12 13 14 15 16 17
18 19 20 21 22 23 24 25 Participants List of competitors Winners Celebrity Kelly Monaco Drew Lachey Emmitt Smith Apolo Anton Ohno Hélio Castroneves Kristi
Yamaguchi Brooke Burke Shawn Johnson Donny Osmond Nicole Scherzinger Jennifer Grey Hines Ward J.R.
</item 5>

<item 6>
Dancing with the Stars (U.S. season 1)
Unsourced material may be challenged and removed . ( August 2015 ) ( Learn how and when to remove this template message ) The six professionals and celebrities
that competed were : Celebrity Notability ( known for ) Professional partner Status Trista Sutter Reality television star Louis Van Amstel Eliminated 1st on
June 8 , 2005 Evander Holyfield Professional boxer Edyta Sliwinska Eliminated 2nd on June 15 , 2005 Rachel Hunter Supermodel & actress Jonathan Roberts
Eliminated 3rd on June 22 , 2005 Joey McIntyre New Kids on the Block singer Ashly DelGrosso Third place on June 29 , 2005 John O'Hurley Actor & game show host
Charlotte Jorgensen Runner - up on July 6 , 2005 Kelly Monaco Actress Alec Mazo Winner on July 6 , 2005 † Kelly Monaco returned for the All - Stars season ,
where this time she was paired with Valentin Chmerkovskiy . They finished third .
</item 6>

<item 7>
Dancing with the Stars (U.S. season 1)
Martinez Donald Driver Melissa Rycroft Kellie Pickler Amber Riley Meryl Davis Alfonso Ribeiro Rumer Willis Bindi Irwin Nyle DiMarco Laurie Hernandez Rashad
Jennings Jordan Fisher Professional Alec Mazo Cheryl Burke Julianne Hough Mark Ballas Derek Hough Kym Johnson Karina Smirnoff Peta Murgatroyd Tony Dovolani
Maksim Chmerkovskiy Witney Carson Valentin Chmerkovskiy Emma Slater Lindsay Arnold Spin - off Dancing with the Stars Junior Games Dancing with the Stars We
Dance ! Retrieved from `` https://en.wikipedia.org/w/index.php?title=Dancing_with_the_Stars_(U.S._season_1)&oldid=809378398 '' Categories : Dancing with the
Stars ( U.S. TV series ) 2005 American television seasons Hidden categories : Television season articles that use custom fields Infobox television season
articles that use the season name parameter Articles needing additional references from August 2015 All articles needing additional references Talk Contents
About Wikipedia Español Français Edit links This page was last edited on 8 November 2017 , at 18 : 55 .
</item 7>

<item 8>
Dancing with the Stars (U.S. season 1)
8 , 9 ) Freestyle `` I Am What I Am '' -- Gloria Gaynor Dance chart ( edit ) Highest scoring dance Lowest scoring dance Couple 5 6 Kelly and Alec Waltz Rumba
Jive Samba Viennese Waltz Foxtrot Paso doble Samba Freestyle John and Charlotte Cha - cha - cha Quickstep Tango Samba Viennese Waltz Foxtrot Paso doble
Quickstep Freestyle Joey and Ashly Cha - cha - cha Quickstep Jive Samba Viennese Waltz Foxtrot Paso doble Rachel and Jonathan Waltz Rumba Tango Samba Viennese
Waltz Evander and Edyta Cha - cha - cha Quickstep Jive Trista and Louis Waltz Rumba References ( edit ) Jump up ^ Associated Press ( 2005 - 06 - 08 ) . `` '
Dancing With the Stars ' Debuts in 2nd Place '' . FoxNews . Retrieved 2009 - 05 - 06 . Jump up ^ Maynard , John ( 2005 - 07 - 13 ) .
</item 8>

<item 9>
Dancing with the Stars (U.S. season 1)
Couples Highest Scoring Dances Lowest Scoring Dances Kelly & Alec Freestyle ( 30 ) Waltz ( 13 ) John & Charlotte Foxtrot , Paso doble , Quickstep & Freestyle (
27 ) Cha - cha - cha ( 20 ) Joey & Ashly Paso doble ( 25 ) Cha - cha - cha , Samba and Foxtrot ( 20 ) Rachel & Jonathan Tango ( 26 ) Waltz ( 20 ) Evander &
Edyta Cha - cha - cha ( 18 ) Jive ( 13 ) Trista & Louis Rumba ( 19 ) Waltz ( 18 ) Weekly scores and songs ( edit ) Unless indicated otherwise , individual
judges scores given in the charts below ( given in parentheses ) are listed in this order from left to right : Carrie Ann Inaba , Len Goodman , Bruno Tonioli .
Week 1 ( edit ) Running order Couple Score Dance Music Joey & Ashly 20 ( 7 , 7 , 6 ) Cha - cha - cha `` Crazy in Love '' -- Beyoncé Rachel & Jonathan 20 ( 7 , 6
,
</item 9>

<item 10>
Dancing with the Stars (U.S. season 1)
The season finale aired July 6 , 2005 . Contents ( hide ) 1 Couples 2 Scoring chart 2.1 Average chart 2.2 Pro 's average chart 3 Highest and lowest scoring
performances 4 Couples ' highest and lowest scoring performances 5 Weekly scores and songs 5.1 Week 1 5.2 Week 2 5.3 Week 3 5.4 Week 4 5.5 Week 5 5.6 Week 6 :
Finale 5.7 Dance off 6 Dance chart 7 References 8 External links Couples ( edit ) This section does not cite any sources . Please help improve this section by
adding citations to reliable sources . Unsourced material may be challenged and removed .
</item 10>
=== END ITEM id=645e50a042847762 ===

=== ITEM id=46dc847a435e9a08 ===
Gold sentences and facts:
Sentence 0: Sports - related costs for the Summer Games since 1960 is on average USD 5.2 billion and for the Winter Games USD 3.1 billion dollars . Severe cost overruns made the 2014 Winter Olympics the most expensive Olympics in history ; with Russian politician Boris Nemtsov citing allegations of corruption among government officials , and Allison Stewart of the Saïd Business School at Oxford citing tight relationships between the government and construction firms . While originally budgeted at US $12 billion , the budget expanded to US $51 billion , surpassing the estimated $44 billion cost of the 2008 Summer Olympics in Beijing , making Sochi the most expensive Olympics in history .
  Fact 0.0: Sports-related costs for the Summer Games since 1960 are on average USD 5.2 billion.
  Fact 0.1: Sports-related costs for the Winter Games are on average USD 3.1 billion.
Sentence 1: Host City Year Final Operating Budget Total Costs Taxpayer Contribution Profit / Loss Year Debt Paid Off Notes Athens Summer Olympics 1896 3,740,000 ₯ Donations by George Averoff of 1,000,000 ₯ covered potential losses London Summer Olympics 1908 US $394,000 est . £ 6,377 Chamonix Winter Olympics 1924 3,500,000 ₣ 2,000,000 ₣ Paris Summer Olympics 1924 5,496,610 ₣ Amsterdam Summer Olympics 1928 $1,183,000 $18,000 St. Moritz Winter Olympics 1928 CHF 706,000 CHF104 , 800 Los Angeles Summer Olympics 1932 $1,000,000 Berlin Summer Olympics 1936 $30,000,000 London Summer Olympics 1948 £ 761,688 £ 29,000 Helsinki Summer Olympics 1952 1,580,000,000 mk 49,000,000 mk Melbourne Summer Olympics 1956 A £ 5,400,000 A £ 300,000 Tokyo Summer Olympics 1964 US $ 72,000,000 Mexico City Summer Olympics 1968 US $ 176,000,000 Munich Summer Olympics 1972 1,972,000,000 DM Montreal Summer Olympics 1976 CDN $ 207,000,000 CDN $ 1,410,000,000 CDN $ 990,000,000 2006 A special tobacco tax was introduced in May 1976 to fund the loss Lake Placid Winter Olympics 1980 US $ 49,000,000 US $ 169,000,000 Moscow Summer Olympics 1980 US $ 231,000,000 US $ 1,350,000,000 USD $1,190,000,000 deficit This was the year that The United States and 64 other Nations boycotted due to the Soviet invasion of Afghanistan . Sarajevo Winter Olympics 1984 US $ 55,400,000 US $ 110,900,000 US $10,000,000 The first Olympics since 1948 to make a profit Los Angeles Summer Olympics 1984 US $ 320,000,000 US $ 413,000,000 $75,000,000 US $250,000,000 1984 The first Summer Olympics since 1932 to make a profit Calgary Winter Olympics 1988 CDN $ 438,000,000 CDN $ 899,000,000 CDN $ 425,000,000 CDN $ 32,000,000 1988 Seoul Summer Olympics 1988 USD $4,000,000,000 US $300,000,000 1988 A record profit for a government - run Olympiad Barcelona Summer Olympics 1992 US $ 850,000,000 US $ 9,300,000,000 US $10,000,000 1992 Operating costs were put at 9.1 % of the total cost . The vast majority of spending was to improve infrastructure . Albertville Winter Olympics 1992 USD $1,200,000,000 on infrastructure USD $67,000,000 Lillehammer Winter Olympics US $ 1,100,000,000 US $ 250,000,000 Atlanta Summer Olympics US $ 1,800,000,000 US $ 609,000,000 US $ 19,000,000 Despite the profit , Atlanta 's heavy reliance on corporate sponsorship drew criticism of commercialization Nagano Winter Olympics 1998 ~ US $ 10,000,000,000 in new infrastructure Net loss Estimated 2015 The full cost of the Nagano Olympics is unknown , due to Nagano Olympic Bid Committee vice-secretary general Sumikazu Yamaguchi ordering accounting documents burned Sydney Summer Olympics 2000 A $ 6,600,000,000 A $ 3,000,000,000 ( A $ 363 , 5000 , 000 borne by the public ) A $ 2,050,000,000 USD $2,100,000,000 loss 2000 Salt Lake City Winter Olympics 2002 US $ 2,000,000,000 US $ 1,200,000,000 US $ 600,000,000 US $ 101,000,000 2002 Additional security costs were incurred in the wake of the September 11 attacks Athens Summer Olympics US $ 15,000,000,000 US $ 9,000,000,000 US $ 6,200,000,000 US $ 14,500,000,000 The cost of the 2004 Athens Summer Games has been cited as a contributor to the Greek government - debt crisis . Many of the venues lie vacant and rotting ; the Independent newspaper reports as many as 21 out of 22 are unused . Turin Winter Olympics 2006 US $ 700,000,000 USD $3,200,000 The Italian government created a lottery game to cover its financial losses . Beijing Summer Olympics 2008 US $ 44,000,000,000 CNY 1,000,000,000 ( US $ 146,000,000 ) 2008 Vancouver Winter Olympics CDN $ 1,700,000,000 ( US $ 1,260,000,000 ) US $ 6,400,000,000 US $ 2,300,000,000 CDN $ 1,900,000 Included in the total US $ 6,400,000,000 cost are the $1,000,000,000 for security , $2,500,000,000 for transportation extensions and upgrades , and $900,000,000 for the new Vancouver Convention Centre ( An additional $554,000,000 was spent by the city including a portion on the Olympic Village ) . London Summer Olympics 2012 US $ 10,400,000,000 US $ 14,600,000,000 US $ 4,400,000,000 GBP £ nil 2012 Additional costs include $90 million for converting the Olympic Stadium ( London ) to a football venue Sochi Winter Olympics 2014 US $ 51,000,000,000 US $ 53,150,000 The most expensive Olympic Games in history , surpassing the previous record set by the 2008 Beijing Summer Olympic Games . Rio de Janeiro Summer Olympics 2016 US $ 13,100,000,000 US $ 11.6 billion Pyeongchang Winter Olympics 2018 US $ 12,900,000,000 Tokyo Summer Olympics 2020 Beijing Winter Olympics 2022 Paris Summer Olympics 2024 TBA 2026 Winter Olympics 2026 Los Angeles Summer Olympics 2028 TBA 2032
  Fact 1.0: The Athens Summer Olympics of 1896 had a final figure of 3,740,000 ₯.
  Fact 1.1: Donations by George Averoff of 1,000,000 ₯ covered potential losses at the Athens Summer Olympics of 1896.
  Fact 1.2: The London Summer Olympics of 1908 had an estimated figure of US $394,000.
  Fact 1.3: The London Summer Olympics of 1908 had a figure of £ 6,377.
  Fact 1.4: The Chamonix Winter Olympics of 1924 had a figure of 3,500,000 ₣.
  Fact 1.5: The Chamonix Winter Olympics of 1924 had a figure of 2,000,000 ₣.
  Fact 1.6: The Paris Summer Olympics of 1924 had a figure of 5,496,610 ₣.
  Fact 1.7: The Amsterdam Summer Olympics of 1928 had figures of $1,183,000 and $18,000.
  Fact 1.8: The St. Moritz Winter Olympics of 1928 had figures of CHF 706,000 and CHF 104,800.
  Fact 1.9: The Los Angeles Summer Olympics of 1932 had a figure of $1,000,000.
  Fact 1.10: The Berlin Summer Olympics of 1936 had a figure of $30,000,000.
  Fact 1.11: The London Summer Olympics of 1948 had figures of £ 761,688 and £ 29,000.
  Fact 1.12: The Helsinki Summer Olympics of 1952 had figures of 1,580,000,000 mk and 49,000,000 mk.
  Fact 1.13: The Melbourne Summer Olympics of 1956 had figures of A £ 5,400,000 and A £ 300,000.
  Fact 1.14: The Tokyo Summer Olympics of 1964 had a figure of US $ 72,000,000.
  Fact 1.15: The Mexico City Summer Olympics of 1968 had a figure of US $ 176,000,000.
  Fact 1.16: The Munich Summer Olympics of 1972 had a figure of 1,972,000,000 DM.
  Fact 1.17: The Montreal Summer Olympics of 1976 had figures of CDN $ 207,000,000, CDN $ 1,410,000,000 and CDN $ 990,000,000.
  Fact 1.18: The Montreal Summer Olympics had its debt paid off in 2006.
  Fact 1.19: A special tobacco tax was introduced in May 1976 to fund the Montreal Olympics loss.
  Fact 1.20: The Lake Placid Winter Olympics of 1980 had figures of US $ 49,000,000 and US $ 169,000,000.
  Fact 1.21: The Moscow Summer Olympics of 1980 had figures of US $ 231,000,000 and US $ 1,350,000,000.
  Fact 1.22: The Moscow Summer Olympics of 1980 had a USD $1,190,000,000 deficit.
  Fact 1.23: The year 1980 was the year the United States and 64 other nations boycotted the Moscow Olympics due to the Soviet invasion of Afghanistan.
  Fact 1.24: The Sarajevo Winter Olympics of 1984 had figures of US $ 55,400,000, US $ 110,900,000 and US $10,000,000.
  Fact 1.25: The Sarajevo Winter Olympics of 1984 was the first Olympics since 1948 to make a profit.
  Fact 1.26: The Los Angeles Summer Olympics of 1984 had figures of US $ 320,000,000, US $ 413,000,000, $75,000,000 and US $250,000,000.
  Fact 1.27: The Los Angeles Summer Olympics of 1984 was the first Summer Olympics since 1932 to make a profit.
  Fact 1.28: The Calgary Winter Olympics of 1988 had figures of CDN $ 438,000,000, CDN $ 899,000,000, CDN $ 425,000,000 and CDN $ 32,000,000.
  Fact 1.29: The Calgary Winter Olympics had its debt paid off in 1988.
  Fact 1.30: The Seoul Summer Olympics of 1988 had figures of USD $4,000,000,000 and US $300,000,000.
  Fact 1.31: The Seoul Summer Olympics of 1988 was a record profit for a government-run Olympiad.
  Fact 1.32: The Barcelona Summer Olympics of 1992 had figures of US $ 850,000,000, US $ 9,300,000,000 and US $10,000,000.
  Fact 1.33: The Barcelona Summer Olympics had its debt paid off in 1992.
  Fact 1.34: Operating costs at the Barcelona Summer Olympics were 9.1 % of the total cost.
  Fact 1.35: The vast majority of Barcelona Summer Olympics spending was to improve infrastructure.
  Fact 1.36: The Albertville Winter Olympics of 1992 had a figure of USD $1,200,000,000 on infrastructure.
  Fact 1.37: The Albertville Winter Olympics of 1992 had a figure of USD $67,000,000.
  Fact 1.38: The Lillehammer Winter Olympics had figures of US $ 1,100,000,000 and US $ 250,000,000.
  Fact 1.39: The Atlanta Summer Olympics had figures of US $ 1,800,000,000, US $ 609,000,000 and US $ 19,000,000.
  Fact 1.40: Despite the profit, Atlanta's heavy reliance on corporate sponsorship drew criticism of commercialization.
  Fact 1.41: The Nagano Winter Olympics of 1998 had about US $ 10,000,000,000 spent on new infrastructure.
  Fact 1.42: The Nagano Winter Olympics had a net loss, with its debt paid off estimated for 2015.
  Fact 1.43: The full cost of the Nagano Olympics is unknown.
  Fact 1.44: The full cost of the Nagano Olympics is unknown because Nagano Olympic Bid Committee vice-secretary general Sumikazu Yamaguchi ordered accounting documents burned.
  Fact 1.45: The Sydney Summer Olympics of 2000 had figures of A $ 6,600,000,000, A $ 3,000,000,000, A $ 363,5000,000 borne by the public, and A $ 2,050,000,000.
  Fact 1.46: The Sydney Summer Olympics of 2000 had a USD $2,100,000,000 loss.
  Fact 1.47: The Sydney Summer Olympics had its debt paid off in 2000.
  Fact 1.48: The Salt Lake City Winter Olympics of 2002 had figures of US $ 2,000,000,000, US $ 1,200,000,000, US $ 600,000,000 and US $ 101,000,000.
  Fact 1.49: The Salt Lake City Winter Olympics had its debt paid off in 2002.
  Fact 1.50: Additional security costs were incurred for the Salt Lake City Winter Olympics in the wake of the September 11 attacks.
  Fact 1.51: The Athens Summer Olympics of 2004 had figures of US $ 15,000,000,000, US $ 9,000,000,000, US $ 6,200,000,000 and US $ 14,500,000,000.
  Fact 1.52: The cost of the 2004 Athens Summer Games has been cited as a contributor to the Greek government debt crisis.
  Fact 1.53: Many of the Athens 2004 venues lie vacant and rotting.
  Fact 1.54: The Independent newspaper reports as many as 21 out of 22 Athens 2004 venues are unused.
  Fact 1.55: The Turin Winter Olympics of 2006 had figures of US $ 700,000,000 and USD $3,200,000.
  Fact 1.56: The Italian government created a lottery game to cover its financial losses from the Turin Winter Olympics.
  Fact 1.57: The Beijing Summer Olympics of 2008 had figures of US $ 44,000,000,000 and CNY 1,000,000,000 (US $ 146,000,000).
  Fact 1.58: The Beijing Summer Olympics had its debt paid off in 2008.
  Fact 1.59: The Vancouver Winter Olympics had figures of CDN $ 1,700,000,000 (US $ 1,260,000,000), US $ 6,400,000,000 and US $ 2,300,000,000.
  Fact 1.60: The Vancouver Winter Olympics row includes the figure CDN $ 1,900,000.
  Fact 1.61: The total US $ 6,400,000,000 cost of the Vancouver Winter Olympics includes $1,000,000,000 for security.
  Fact 1.62: The total US $ 6,400,000,000 cost of the Vancouver Winter Olympics includes $2,500,000,000 for transportation extensions and upgrades.
  Fact 1.63: The total US $ 6,400,000,000 cost of the Vancouver Winter Olympics includes $900,000,000 for the new Vancouver Convention Centre.
  Fact 1.64: An additional $554,000,000 was spent by the city of Vancouver, including a portion on the Olympic Village.
  Fact 1.65: The London Summer Olympics of 2012 had figures of US $ 10,400,000,000, US $ 14,600,000,000 and US $ 4,400,000,000.
  Fact 1.66: The London Summer Olympics of 2012 had GBP £ nil listed.
  Fact 1.67: Additional costs of the London 2012 Olympics include $90 million for converting the Olympic Stadium (London) to a football venue.
  Fact 1.68: The Sochi Winter Olympics of 2014 had figures of US $ 51,000,000,000 and US $ 53,150,000.
  Fact 1.69: The Sochi Winter Olympics of 2014 were the most expensive Olympic Games in history.
  Fact 1.70: The Sochi Winter Olympics of 2014 surpassed the previous record set by the 2008 Beijing Summer Olympic Games.
  Fact 1.71: The Rio de Janeiro Summer Olympics of 2016 had figures of US $ 13,100,000,000 and US $ 11.6 billion.
  Fact 1.72: The Pyeongchang Winter Olympics of 2018 had a figure of US $ 12,900,000,000.
  Fact 1.73: The table lists the Tokyo Summer Olympics of 2020, the Beijing Winter Olympics of 2022, the Paris Summer Olympics of 2024 and the 2026 Winter Olympics.
  Fact 1.74: The table contains TBA entries alongside the 2020 to 2032 Games.
  Fact 1.75: The table lists the Los Angeles Summer Olympics of 2028 and the 2032 Olympics.

Retrieved items:
<item 1>
Cost of the Olympic Games
Jump up ^ `` Olympic Cities : Booms and Busts '' . Cnbc.com. 2012 - 01 - 19 . Retrieved 2014 - 02 - 07 . Jump up ^ `` Turin Winter Olympics 2006 - 2014 - 02 -
06 - CIPRA : Living in the Alps '' . Retrieved 2014 - 02 - 12 . Jump up ^ `` Olympic Cities : Booms and Busts '' . Cnbc.com. 2012 - 01 - 19 . Retrieved 2014 -
02 - 07 . Jump up ^ `` Beijing Olympics to cost China 44,000,000,000 dollars - 2008 - 06 - 08 - Pravda News '' . Retrieved 2014 - 02 - 12 . Jump up ^ `` Beijing
Olympics ' profit exceeds 1b yuan '' . Jump up ^ `` Olympics cost B.C. $925 M '' . CBC Sports . July 9 , 2010 . Retrieved February 4 , 2013 .
</item 1>

<item 2>
Cost of the Olympic Games
000,000,000 US $ 1,200,000,000 US $ 600,000,000 US $ 101,000,000 2002 Additional security costs were incurred in the wake of the September 11 attacks Athens
Summer Olympics US $ 15,000,000,000 US $ 9,000,000,000 US $ 6,200,000,000 US $ 14,500,000,000 The cost of the 2004 Athens Summer Games has been cited as a
contributor to the Greek government - debt crisis . Many of the venues lie vacant and rotting ; the Independent newspaper reports as many as 21 out of 22 are
unused . Turin Winter Olympics 2006 US $ 700,000,000 USD $3,200,000 The Italian government created a lottery game to cover its financial losses .
</item 2>

<item 3>
Cost of the Olympic Games
London Summer Olympics 2012 US $ 10,400,000,000 US $ 14,600,000,000 US $ 4,400,000,000 GBP £ nil 2012 Additional costs include $90 million for converting the
Olympic Stadium ( London ) to a football venue Sochi Winter Olympics 2014 US $ 51,000,000,000 US $ 53,150,000 The most expensive Olympic Games in history ,
surpassing the previous record set by the 2008 Beijing Summer Olympic Games . Rio de Janeiro Summer Olympics 2016 US $ 13,100,000,000 US $ 11.6 billion
Pyeongchang Winter Olympics 2018 US $ 12,900,000,000 Tokyo Summer Olympics 2020 Beijing Winter Olympics 2022 Paris Summer Olympics 2024 TBA 2026 Winter Olympics
2026 Los Angeles Summer Olympics 2028 TBA 2032 See also ( edit ) List of megaprojects References ( edit ) Jump up ^ Bennetts , Marc ( 19 January 2014 ) .
</item 3>

<item 4>
Cost of the Olympic Games
Cost of the Olympic Games - wikipedia Cost of the Olympic Games Jump to : navigation , search This article possibly contains original research . Please improve
it by verifying the claims made and adding inline citations . Statements consisting only of original research should be removed . ( February 2018 ) ( Learn how
and when to remove this template message ) Sports - related costs for the Summer Games since 1960 is on average USD 5.2 billion and for the Winter Games USD 3.1
billion dollars . Severe cost overruns made the 2014 Winter Olympics the most expensive Olympics in history ; with Russian politician Boris Nemtsov citing
allegations of corruption among government officials , and Allison Stewart of the Saïd Business School at Oxford citing tight relationships between the
government and construction firms . While originally budgeted at US $12 billion , the budget expanded to US $51 billion , surpassing the estimated $44 billion
cost of the 2008 Summer Olympics in Beijing , making Sochi the most expensive Olympics in history . The table below is listing the not so hard costs of the
Games .
</item 4>

<item 5>
Cost of the Olympic Games
2002 - 09 - 18 . Retrieved 2010 - 02 - 13 . Jump up ^ `` As Olympic glow fades , Athens Questions $15,000,000,000 Cost - 2008 - 07 - 21 - The Christian Science
Monitor '' . Retrieved 2010 - 02 - 13 . Jump up ^ `` How the 2004 Olympics Triggered Greece 's Decline - 2012 - 08 - 02 - Bloomberg Business '' . Retrieved 2014
- 02 - 12 . Jump up ^ `` Olympics ' may cost Greece dear ' - 2004 - 06 - 02 - BBC NEWS '' . BBC News . 2004 - 06 - 02 . Retrieved 2014 - 02 - 12 . Jump up ^
Schlotterbeck , Bianca ( 19 January 2012 ) . `` Athens , Greece ( Summer , 2004 ) '' . Olympic Cities : Booms and Busts . CNBC . Retrieved 13 February 2013 .
Jump up ^ `` Olympic Cities : Booms and Busts '' .
</item 5>

<item 6>
Cost of the Olympic Games
Albertville Winter Olympics 1992 USD $1,200,000,000 on infrastructure USD $67,000,000 Lillehammer Winter Olympics US $ 1,100,000,000 US $ 250,000,000 Atlanta
Summer Olympics US $ 1,800,000,000 US $ 609,000,000 US $ 19,000,000 Despite the profit , Atlanta 's heavy reliance on corporate sponsorship drew criticism of
commercialization Nagano Winter Olympics 1998 ~ US $ 10,000,000,000 in new infrastructure Net loss Estimated 2015 The full cost of the Nagano Olympics is
unknown , due to Nagano Olympic Bid Committee vice-secretary general Sumikazu Yamaguchi ordering accounting documents burned Sydney Summer Olympics 2000 A $
6,600,000,000 A $ 3,000,000,000 ( A $ 363 , 5000 , 000 borne by the public ) A $ 2,050,000,000 USD $2,100,000,000 loss 2000 Salt Lake City Winter Olympics 2002
US $ 2,000,000,000 US $ 1,200,000,
</item 6>

<item 7>
Cost of the Olympic Games
Retrieved 2014 - 02 - 07 . ^ Jump up to : Nina Berglund . `` Lillehammer marks OL anniversary '' . News In English . Retrieved February 7 , 2014 . compared to
the NOK 8,500,000,000 spent on Lillehammer ^ Jump up to : `` Exchange rate , US dollar ( USD ) '' . Norges Bank . Retrieved February 7 , 2014 . January 1994
7.5109 ^ Jump up to : Schlotterbeck , Bianca ( 19 January 2012 ) . `` Nagano , Japan ( Winter 1998 ) '' . Olympic Cities : Booms and Busts . CNBC . Retrieved 13
February 2013 . Jump up ^ Craig , David ( 2009 ) . Squandered . Constable & Robinson Limited . p. 300 . ISBN 9781849011617 . Jump up ^ `` Olympic glory at any
price ? - John Madden & James Giesecke - 26 Jul 2012 , Business Spectator '' .
</item 7>

<item 8>
Cost of the Olympic Games
Retrieved 2014 - 02 - 12 . Jump up ^ `` Sydney 2000 - Auditor Slams Costs '' . Archived from the original on 2005 - 02 - 07 . Jump up ^ `` Hosting the Olympics
: cash cow or money pit ? '' . Retrieved 2012 - 12 - 17 . Jump up ^ Roberts , Selena ( 2002 - 02 - 04 ) . `` IOC 's Rogge Steps into the Cold - Feb 4 , 2002 -
The New York Times '' . Retrieved 2010 - 02 - 13 . Jump up ^ `` Olympics budget nearly $2,000,000,000 - Dec 11 , 2001 - Sports Illustrated '' . CNN . 2001 - 12
- 11 . Retrieved 2014 - 02 - 12 . Jump up ^ `` Mitt Romney 's Olympic history in the spotlight '' . The Daily Telegraph ( Sydney ) . 28 July 2012 . Jump up ^ ``
Salt Lake Tops Forecast - 2002 - 09 - 18 - The New York Times '' .
</item 8>

<item 9>
Cost of the Olympic Games
International Olympic Committee . Retrieved 2014 - 02 - 13 . ^ Jump up to : `` Official Report of the XXII Olympiad Moscow 1980 '' ( PDF ) . International
Olympic Committee . 1981 . Retrieved 2014 - 02 - 13 . ^ Jump up to : `` FINAL REPORT published by the Organising Committee of the XlVth Winter Olympic Games
1984 at Sarajevo '' ( PDF ) . ^ Jump up to : `` Official Report of the Games of the XXIIIrd Olympiad Los Angeles , 1984 '' ( PDF ) . International Olympic
Committee . 1985 . Retrieved 2014 - 02 - 13 . ^ Jump up to : Julia Campbell . `` Cost to Host Olympic Games Skyrockets '' . ABC News . Retrieved August 6 , 2013
. ^ Jump up to : `` Olympic Cities : Booms and Busts '' . Cnbc.com. 2012 - 01 - 19 . Retrieved 2014 - 02 - 07 .
</item 9>

<item 10>
Cost of the Olympic Games
The table below is listing the not so hard costs of the Games . Due to the multitude of reporting methods , the table contains both the operating costs and
total final costs ( which include various infrastructure upgrades and security costs ) , known and not estimated . Net loss or gain are measured against the
operating budgets . Intangible costs ( such as to the environment and society ) and not so total benefits ( through tourism ) are not included here . Contents (
hide ) 1 Table 2 See also 3 References 4 External links Table ( edit ) Host City Year Final Operating Budget Total Costs Taxpayer Contribution Profit / Loss
Year Debt Paid Off Notes Athens Summer Olympics 1896 3,740,000 ₯ Donations by George Averoff of 1,000,000 ₯ covered potential losses London Summer Olympics 1908
US $394,000 est . £ 6,377 Chamonix Winter Olympics 1924 3,500,000 ₣ 2,000,000 ₣ Paris Summer Olympics 1924 5,496,610 ₣ Amsterdam Summer Olympics 1928 $1,183,000
$18,000 St.
</item 10>
=== END ITEM id=46dc847a435e9a08 ===

=== ITEM id=a54cc5f092fd95e2 ===
Gold sentences and facts:
Sentence 0: The Second Great Awakening occurred in several episodes and over different denominations ; however , the revivals were very similar . As the most effective form of evangelizing during this period , revival meetings cut across geographical boundaries , and the movement quickly spread throughout Kentucky , Tennessee and southern Ohio . Each denomination had assets that allowed it to thrive on the frontier . The Methodists had an efficient organization that depended on itinerant ministers , known as circuit riders , who sought out people in remote frontier locations . The circuit riders came from among the common people , which helped them establish rapport with the frontier families they hoped to convert .
  Fact 0.0: The Second Great Awakening occurred in several episodes.
  Fact 0.1: The Second Great Awakening occurred over different denominations.
  Fact 0.2: The revivals of the Second Great Awakening were very similar to each other.
  Fact 0.3: Revival meetings were the most effective form of evangelizing during this period.
  Fact 0.4: Revival meetings cut across geographical boundaries.
  Fact 0.5: The revival movement quickly spread throughout Kentucky, Tennessee and southern Ohio.
  Fact 0.6: Each denomination had assets that allowed it to thrive on the frontier.
  Fact 0.7: The Methodists had an efficient organization.
  Fact 0.8: The Methodist organization depended on itinerant ministers known as circuit riders.
  Fact 0.9: Circuit riders sought out people in remote frontier locations.
  Fact 0.10: Circuit riders came from among the common people.
  Fact 0.11: Coming from among the common people helped circuit riders establish rapport with the frontier families they hoped to convert.
Sentence 1: The revivals enrolled millions of new members in existing evangelical denominations and led to the formation of new denominations . Many converts believed that the Awakening heralded a new millennial age . The Second Great Awakening stimulated the establishment of many reform movements designed to remedy the evils of society before the anticipated Second Coming of Jesus Christ .
  Fact 1.0: The revivals enrolled millions of new members in existing evangelical denominations.
  Fact 1.1: The revivals led to the formation of new denominations.
  Fact 1.2: Many converts believed that the Awakening heralded a new millennial age.
  Fact 1.3: The Second Great Awakening stimulated the establishment of many reform movements.
  Fact 1.4: The reform movements were designed to remedy the evils of society before the anticipated Second Coming of Jesus Christ.
Sentence 2: In the newly settled frontier regions , the revival was implemented through camp meetings . These often provided the first encounter for some settlers with organized religion , and they were important as social venues . The camp meeting was a religious service of several days ' length with preachers . Settlers in thinly populated areas gathered at the camp meeting for fellowship as well as worship . The sheer exhilaration of participating in a religious revival with crowds of hundreds and perhaps thousands of people inspired the dancing , shouting , and singing associated with these events . The revivals followed an arc of great emotional power , with an emphasis of the individual 's sins and need to turn to Christ , restored by a sense of personal salvation . Upon their return home , most converts joined or created small local churches , which grew rapidly . The Second Great Awakening marked a religious transition in society in America . Many Americans from the Calvinist sect emphasized man 's inability to save themselves and that their only way to be saved was from grace from God .
  Fact 2.0: In the newly settled frontier regions, the revival was implemented through camp meetings.
  Fact 2.1: Camp meetings often provided the first encounter for some settlers with organized religion.
  Fact 2.2: Camp meetings were important as social venues.
  Fact 2.3: A camp meeting was a religious service of several days' length with preachers.
  Fact 2.4: Settlers in thinly populated areas gathered at the camp meeting for fellowship as well as worship.
  Fact 2.5: Participating in a religious revival with crowds of hundreds and perhaps thousands of people was sheer exhilaration.
  Fact 2.6: That exhilaration inspired the dancing, shouting, and singing associated with camp meeting events.
  Fact 2.7: The revivals followed an arc of great emotional power.
  Fact 2.8: The revivals emphasized the individual's sins and need to turn to Christ.
  Fact 2.9: The revivals were restored by a sense of personal salvation.
  Fact 2.10: Upon their return home, most converts joined or created small local churches.
  Fact 2.11: The small local churches grew rapidly.
  Fact 2.12: The Second Great Awakening marked a religious transition in society in America.
  Fact 2.13: Many Americans from the Calvinist sect emphasized man's inability to save themselves.
  Fact 2.14: Many Americans from the Calvinist sect emphasized that their only way to be saved was by grace from God.

Retrieved items:
<item 1>
Second Great Awakening
a re-emphasis on Wesleyan teachings on sanctification emerged during the Second Great Awakening , leading to a distinction between Mainline Methodism and
Holiness churches . Restoration movement ( edit ) Main article : Restoration Movement See also : Restorationism The idea of restoring a `` primitive '' form of
Christianity grew in popularity in the U.S. after the American Revolution . This desire to restore a purer form of Christianity without an elaborate hierarchy
contributed to the development of many groups during the Second Great Awakening , including the Mormons , Baptists and Shakers . Several factors made the
restoration sentiment particularly appealing during this time period : To immigrants in the early 19th century , the land in the United States seemed pristine ,
edenic and undefiled -- `` the perfect place to recover pure , uncorrupted and original Christianity '' -- and the tradition - bound European churches seemed
out of place in this new setting . A primitive faith based on the Bible alone promised a way to sidestep the competing claims of the many denominations
available and for congregations to find assurance of being right without the security of an established national church . The Restoration Movement began during
, and was greatly influenced by , the Second Great Awakening .
</item 1>

<item 2>
Second Great Awakening
West ( edit ) Main article : Revival of 1800 In the newly settled frontier regions , the revival was implemented through camp meetings . These often provided
the first encounter for some settlers with organized religion , and they were important as social venues . The camp meeting was a religious service of several
days ' length with preachers . Settlers in thinly populated areas gathered at the camp meeting for fellowship as well as worship . The sheer exhilaration of
participating in a religious revival with crowds of hundreds and perhaps thousands of people inspired the dancing , shouting , and singing associated with these
events . The revivals followed an arc of great emotional power , with an emphasis of the individual 's sins and need to turn to Christ , restored by a sense of
personal salvation . Upon their return home , most converts joined or created small local churches , which grew rapidly . The Second Great Awakening marked a
religious transition in society in America . Many Americans from the Calvinist sect emphasized man 's inability to save themselves and that their only way to be
saved was from grace from God . The Revival of 1800 in Logan County , Kentucky , began as a traditional Presbyterian sacramental occasion .
</item 2>

<item 3>
Second Great Awakening
Contents ( hide ) 1 Spread of revivals 1.1 Background 1.2 Theology 1.3 Burned - over district 1.4 West and Tidewater South 1.5 West 1.6 Church membership soars
2 Subgroups 2.1 Adventism 2.2 Holiness movement 2.3 Restoration Movement 3 Culture and society 4 Slaves and free Africans 5 Women 6 Prominent figures 7
Political implications 8 See also 9 References 10 Further reading 10.1 Historiography Spread of revivals ( edit ) Part of a series on Baptists Background ( show
) Christianity Protestantism Puritanism Anabaptism Doctrine ( show ) Priesthood of all believers Individual soul liberty Separation of church and state Sola
scriptura Congregationalism Ordinances Offices Confessions Key figures ( show ) John Smyth Thomas Helwys Roger Williams John Clarke John Bunyan Shubal Stearns
Andrew Fuller Charles Spurgeon D.N. Jackson James Robinson Graves William Bullein Johnson William Carey Luther Rice Martin Luther King Jr .
</item 3>

<item 4>
Second Great Awakening
Many converts believed that the Awakening heralded a new millennial age . The Second Great Awakening stimulated the establishment of many reform movements
designed to remedy the evils of society before the anticipated Second Coming of Jesus Christ . Historians named the Second Great Awakening in the context of the
First Great Awakening of the 1730s and 1740s and of the Third Great Awakening of the late 1850s to early 1900s . These revivals were part of a much larger
Romantic religious movement that was sweeping across Europe at the time , mainly throughout England , Scotland , and Germany .
</item 4>

<item 5>
Second Great Awakening
See Uncommon Christian Ministries Long , Kimberly Bracken . `` The Communion Sermons of James Mcgready : Sacramental Theology and Scots - Irish Piety on the
Kentucky Frontier '' , Journal of Presbyterian History , 2002 80 ( 1 ) : 3 -- 16 . ISSN 0022 - 3883 . JSTOR 23336302 . Loveland Anne C. Southern Evangelicals
and the Social Order , 1800 -- 1860 , ( 1980 ) McLoughlin William G. Modern Revivalism , 1959 . McLoughlin William G. Revivals , Awakenings , and Reform : An
Essay on Religion and Social Change in America , 1607 -- 1977 , 1978 . Marsden , George M. The Evangelical Mind and the New School Presbyterian Experience : A
Case Study of Thought and Theology in Nineteenth - Century America ( 1970 ) . Meyer , Neil . `` Falling for the Lord : Shame , Revivalism , and the Origins of
the Second Great Awakening . ''
</item 5>

<item 6>
Second Great Awakening
the Second reflected Romanticism characterized by enthusiasm , emotion , and an appeal to the super-natural . It rejected the skepticism , deism , and
rationalism left over from the Enlightenment . At about the same time , similar movements flourished in Europe . Pietism was sweeping German countries .
Evangelicalism was waxing strong in England . The Second Great Awakening occurred in several episodes and over different denominations ; however , the revivals
were very similar . As the most effective form of evangelizing during this period , revival meetings cut across geographical boundaries , and the movement
quickly spread throughout Kentucky , Tennessee and southern Ohio . Each denomination had assets that allowed it to thrive on the frontier . The Methodists had
an efficient organization that depended on itinerant ministers , known as circuit riders , who sought out people in remote frontier locations . The circuit
riders came from among the common people , which helped them establish rapport with the frontier families they hoped to convert . Theology ( edit ) Main article
: Postmillennialism Postmillennialism theology dominated American Protestantism in the first half of the 19th century .
</item 6>

<item 7>
Second Great Awakening
While Protestant religion had previously played an important role on the American political scene , the Second Great Awakening strengthened the role it would
play . See also ( edit ) Baptist portal Methodism portal Calvinism portal Seventh - day Adventist Church portal Latter Day Saints portal LDS Church portal
Advent Christian Church Christian revival Christianity in the 19th century The Church of Jesus Christ of Latter - day Saints Cumberland Presbyterian Church
Ethnocultural politics in the United States Holiness movement Restoration Movement Seventh - day Adventist Church References ( edit ) ^ Jump up to : Timothy L.
Smith , Revivalism and Social Reform : American Protestantism on the Eve of the Civil War ( 1957 ) . Jump up ^ Heyrman , Christine Leigh . `` The First Great
Awakening '' . Divining America , TeacherServe . National Humanities Center . Jump up ^ Henry B. Clark ( 1982 ) . Freedom of Religion in America : Historical
Roots , Philosophical Concepts , Contemporary Problems . Transaction Publishers . p. 16 . ^ Jump up to : Nancy Cott , `` Young Women in the Second Great
Awakening in New England , '' Feminist Studies , ( 1975 ) , 3 # 1 p.
</item 7>

<item 8>
Second Great Awakening
The Restoration Movement began during , and was greatly influenced by , the Second Great Awakening . While the leaders of one of the two primary groups making
up this movement , Thomas Campbell and Alexander Campbell , resisted what they saw as the spiritual manipulation of the camp meetings , the revivals contributed
to the development of the other major branch , led by Barton W. Stone . The Southern phase of the Awakening `` was an important matrix of Barton Stone 's reform
movement '' and shaped the evangelistic techniques used by both Stone and the Campbells . Culture and society ( edit ) Efforts to apply Christian teaching to
the resolution of social problems presaged the Social Gospel of the late 19th century . Converts were taught that to achieve salvation they needed not just to
repent personal sin but also work for the moral perfection of society , which meant eradicating sin in all its forms . Thus , evangelical converts were leading
figures in a variety of 19th century reform movements . Congregationalists set up missionary societies to evangelize the western territory of the northern tier
. Members of these groups acted as apostles for the faith , and also as educators and exponents of northeastern urban culture .
</item 8>

<item 9>
Second Great Awakening
The Second Great Awakening served as an `` organizing process '' that created `` a religious and educational infrastructure '' across the western frontier that
encompassed social networks , a religious journalism that provided mass communication , and church - related colleges . Publication and education societies
promoted Christian education ; most notable among them was the American Bible Society , founded in 1816 . Women made up a large part of these voluntary
societies . The Female Missionary Society and the Maternal Association , both active in Utica , NY , were highly organized and financially sophisticated women
's organizations responsible for many of the evangelical converts of the New York frontier . There were also societies that broadened their focus from
traditional religious concerns to larger societal ones . These organizations were primarily sponsored by affluent women . They did not stem entirely from the
Second Great Awakening , but the revivalist doctrine and the expectation that one 's conversion would lead to personal action accelerated the role of women 's
social benevolence work . Social activism influenced abolition groups and supporters of the Temperance movement . They began efforts to reform prisons and care
for the handicapped and mentally ill . They believed in the perfectibility of people and were highly moralistic in their endeavors .
</item 9>

<item 10>
Second Great Awakening
Second Great Awakening - wikipedia Second Great Awakening Jump to : navigation , search An engraving of a Methodist camp meeting in 1819 ( Library of Congress )
. Great Awakening First ( c. 1730 -- 1755 ) Second ( c. 1790 -- 1840 ) Third ( c. 1855 -- 1900 ) Fourth ( c. 1960 -- 1980 ) The Second Great Awakening was a
Protestant religious revival during the early 19th century in the United States . The movement began around 1790 , gained momentum by 1800 and , after 1820 ,
membership rose rapidly among Baptist and Methodist congregations whose preachers led the movement . It was past its peak by the late 1850s . The Second Great
Awakening reflected Romanticism characterized by enthusiasm , emotion , and an appeal to the super-natural . It rejected the skeptical rationalism and deism of
the Enlightenment . The revivals enrolled millions of new members in existing evangelical denominations and led to the formation of new denominations . Many
converts believed that the Awakening heralded a new millennial age .
</item 10>
=== END ITEM id=a54cc5f092fd95e2 ===

=== ITEM id=3ee02783fd7486e4 ===
Gold sentences and facts:
Sentence 0: The Rhine -- Herne Canal ( German : Rhein - Herne - Kanal ) is a 45.6 - kilometre - long ( 28.3 mi ) transportation canal in the Ruhr area of North Rhine - Westphalia , Germany , with five canal locks . The canal was built over a period of eight years ( 5 April 1906 - 14 July 1914 ) and connects the harbour in Duisburg on the Rhine ( 51 ° 26 ′ 59 '' N 6 ° 46 ′ 1 '' E ﻿ / ﻿ 51.44972 ° N 6.76694 ° E ﻿ / 51.44972 ; 6.76694 ) with the Dortmund - Ems Canal near Henrichenburg ( 51 ° 37 ′ 1 '' N 7 ° 19 ′ 19 '' E ﻿ / ﻿ 51.61694 ° N 7.32194 ° E ﻿ / 51.61694 ; 7.32194 ) , following the valley of the Emscher . It was widened in the 1980s . The Rhein - Herne canal ship was designed specifically for this canal ; normally of about 1300 -- 1350 ton capacity , it has a maximum draft of 2.50 metres ( 8.2 ft ) , a length of approximately 80 metres ( 260 ft ) , and maximum beam of 9.50 metres ( 31.2 ft ) .
  Fact 0.0: The Rhine-Herne Canal (German: Rhein-Herne-Kanal) is a transportation canal.
  Fact 0.1: The Rhine-Herne Canal is 45.6 kilometres long (28.3 miles).
  Fact 0.2: The Rhine-Herne Canal is in the Ruhr area of North Rhine-Westphalia, Germany.
  Fact 0.3: The Rhine-Herne Canal has five canal locks.
  Fact 0.4: The canal was built over a period of eight years, from 5 April 1906 to 14 July 1914.
  Fact 0.5: The canal connects the harbour in Duisburg on the Rhine with the Dortmund-Ems Canal near Henrichenburg.
  Fact 0.6: The Duisburg harbour is at coordinates 51°26′59″N 6°46′1″E.
  Fact 0.7: The Henrichenburg end is at coordinates 51°37′1″N 7°19′19″E.
  Fact 0.8: The canal follows the valley of the Emscher.
  Fact 0.9: The canal was widened in the 1980s.
  Fact 0.10: The Rhein-Herne canal ship was designed specifically for this canal.
  Fact 0.11: The Rhein-Herne canal ship normally has a capacity of about 1300 to 1350 tons.
  Fact 0.12: The Rhein-Herne canal ship has a maximum draft of 2.50 metres (8.2 ft).
  Fact 0.13: The Rhein-Herne canal ship has a length of approximately 80 metres (260 ft).
  Fact 0.14: The Rhein-Herne canal ship has a maximum beam of 9.50 metres (31.2 ft).

Retrieved items:
<item 1>
Rhine–Herne Canal
Ports Wanne west and Wanne east ( edit ) See also : Wanne - Herner Eisenbahn und Hafen Like many other the port of Westhafen Wanne ( Wanne west harbour ) was
originally built for the mutual benefit of the mines and canal , with a railway connecting the two . Today the quayside still has coal traffic and associated
industries . The docks of the port at Wanner West were filled in to make room for the Güterverkehrszentrum Emscher - a logistics ( road , rail and river )
centre working in association with other companies . The Wanne - Herner Eisenbahn und Hafen GmbH ( WHE ) operates both the remaining canal side docks and the
associated railway . The Osthafen Wanne ( Port Wanne east ) are also operated by the WHE ; the ports and rail link serves a major scrap metal works . City port
of Recklinghausen ( edit ) Recklinghausen port ( 2008 ) grain terminal Originally the city of Recklinghausen had no access to the canal ; through an exchange of
land with the city of Herne the city gained access to the canal .
</item 1>

<item 2>
Rhine–Herne Canal
Contents ( hide ) 1 Water system 2 Locks 2.1 Duisburg - Meiderich sluicegate 3 Ports 3.1 Duisburg 3.2 Oberhausen marina 3.3 City port of Essen 3.4 City port of
Gelsenkirchen 3.5 Port Grimberg 3.6 Ports Wanne west and Wanne east 3.7 City port of Recklinghausen 3.8 Port of the Frederick the Great mine 3.9 Port of the
King Ludwig mine 3.10 Port Victor 3.11 Port Rütgers 3.12 Port Luck 3.13 Port Marmorit 3.14 Waltrop lock park 4 Ecosystem 5 Cities 6 References and notes Water
system ( edit ) The distribution of water in the canal is realised through five pump stations . Located at Duisburg - Meiderich , Oberhausen , Gelsenkirchen ,
Wanne - Eickel and East Herne , they pump water from the Rhine into the canal at each stage .
</item 2>

<item 3>
Rhine–Herne Canal
From here maintenance and repair work are organised and performed . Oberhausen marina ( edit ) The Oberhausen marina is a yacht port opened in 2004 in close
proximity to the CentrO shopping centre . The port has a water depth of 2.5 metres ( 8.2 ft ) and over 13 bridges . Also at the Oberhausen marina are the
attractions Sealife Oberhausen and Oberhausen Modellbahn World . City port of Essen ( edit ) The Stadthafen Essen ( City port of Essen ) is the youngest port on
the Rhine -- Herne Canal opened in 1934 by the city of Essen , and operated since 1987 by Stadtwerke Essen .
</item 3>

<item 4>
Rhine–Herne Canal
The main product dealt with here is pitch obtained from coal tar from the refining of coal . The port has a Pneumatic barrier . Port Luck ( edit ) Port Luck
Port Luck ( Hafen Luck ) belongs to the construction and building materials company Luck located on the east bank , a little more than 2 kilometres upstream of
Rütgers Chemicals . Port Marmorit ( edit ) Port marmorit with its single fixed crane Port Marmorit ( Hafen Marmorit ) is the port of cement and lime mortar
manufacturer and distributor Marmorit Knauf GmbH located north of a disused canal side channel and just north of the bridge carrying Wartburgstraße (
Wartburgstreet ) . The actual facility consists of little more than a pier . Waltrop lock park ( edit ) Here after 45.6 kilometres ( 28.3 mi ) ends ( or begins
) the Rhine -- Herne Canal , which meets the Dortmund - Ems canal ( at the 14.7 - kilometre mark ) near the old Henrichenburg boat lift ( replaced in 1962 by a
newer lock , and in 1989 by a new larger lock ) .
</item 4>

<item 5>
Rhine–Herne Canal
The industrial port is ~ 900 by 60 metres ( 2,950 by 200 ft ) and the trading port ~ 500 by 75 metres ( 1,640 by 246 ft ) . A private railway of length 18.4
kilometres ( 11.4 mi ) is operated which is linked to the tracks of the Deutsche Bahn Formerly the port dealt mainly with coal and mining products , now the
port is a major hub for petroleum products and second - largest inland port for cereals in North Rhine - Westphalia . It is operated by the Gelsenkirchener
Logistik - , Hafen - und Servicegesellschaft ( Gelsenkirchener logistics , harbour and service company . ) a subsidiary of Stadtwerke Gelsenkirchen . Port
Grimberg ( edit ) The port at Grimberg Hafen Grimberg ( Port Grimberg ) is on the south canal bank in Gelsenkirchen , near the city boundary to Herne . It was
named after the nearby Castle Grimberg , of which today only the foundations remain .
</item 5>

<item 6>
Rhine–Herne Canal
A possible reconstruction of the sluice gate was commissioned in 1914 with the canal opening . Instead of the old sluice of 165 metres ( 541 ft ) in length and
10 metres ( 33 ft ) in breadth a new sluice chamber with a usable length of 190 metres ( 620 ft ) and a breadth of 12 metres ( 39 ft ) were commissioned , so
that bigger ships with lighters ( barges ) as specified in Euro class II can pass through the sluice gates . Ports ( edit ) Duisburg ( edit ) Main article :
Duisburg § Duisburg Port The canal begins in Duisburg along with the Schifffahrtskanal ; connected to land by three bays of the Kanalhafen , and to the rest of
the world via the Rhine . The Rhine - Herne Canal headquarters is in the Meiderich district of Duisburg , it serves as the water and shipping management canal
and its locks from Essen to Herne . From here maintenance and repair work are organised and performed .
</item 6>

<item 7>
Rhine–Herne Canal
The port consists of a doubled sided canal side quay together with a harbour port at right angles onto the canal , which together form a water area of 63,000
square metres ( 680,000 sq ft ) with associated dock - land of 300,000 square metres ( 3,200,000 sq ft ) The dock handles bulk goods such as solid mineral fuels
, petroleum and chemical products , stone , earth and iron and steel as well as recyclable materials , and is an important transhipment point . City port of
Gelsenkirchen ( edit ) The port of Gelsenkirchen The port of the city of Gelsenkirchen ( Stadthafen Gelsenkirchen ) was opened in July 1914 and is the oldest
port on the Rhine -- Herne Canal . The port has two docks branched from the canal situated just east of the lock at Gelsenkirchen and has a water surface of
117,800 square metres ( 1,268,000 sq ft ) .
</item 7>

<item 8>
Rhine–Herne Canal
Jump up ^ Stadthafen Essen route-industriekultur.de Jump up ^ Hafen Gelsenkirchen hafen-ge.de Jump up ^ Stadthafen Gelsenkirchen route-industriekultur.de Jump
up ^ Hafen Grimberg und Erzbahn route-industrielkultur.de ^ Jump up to : Westhafen Wanne is more clearly on the north bank of the canal , with Osthafen Wanne
almost directly opposite on the south bank . Both are just east of the Schleuse Wanne - Eickel ( Wanne - Eickel lock ) Jump up ^ In German the port is named
Stadthafen Recklinghausen ^ Jump up to : Aerial image of Hafen der Zeche König Ludwig , ' Hafen der Zeche Friedrich der Große and Schleuse Herne - Ost
multimap.com ^ Jump up to : Aerial image of Hafen der Zeche König Ludwig , ' Hafen der Zeche Friedrich der Große and Schleuse Herne - Ost maps.google.co.uk Jump
up ^ Hafen Victor route-industriekultur.
</item 8>

<item 9>
Rhine–Herne Canal
Rhine -- Herne canal - wikipedia Rhine -- Herne canal Jump to : navigation , search View from the Gasometer Oberhausen The Rhine -- Herne Canal ( German : Rhein
- Herne - Kanal ) is a 45.6 - kilometre - long ( 28.3 mi ) transportation canal in the Ruhr area of North Rhine - Westphalia , Germany , with five canal locks .
</item 9>

<item 10>
Rhine–Herne Canal
The terminal is used as a mooring for pleasure craft . It forms part of the Waltrop lock park . Ecosystem ( edit ) Fish that inhabit the canal include European
eel , European carp , zander , carp bream , common roach and common rudd , as well as the less common pike and rainbow trout . Cities ( edit ) Duisburg
Oberhausen Essen Bottrop Gelsenkirchen Herne Recklinghausen Castrop - Rauxel Waltrop Datteln Wikimedia Commons has media related to Rhein - Herne - Kanal .
References and notes ( edit ) Jump up ^ Route A42 of the Federal Motorways now partly follows the route of the disused canal , ie . the section from the port at
the Friedrich der Große coal mine to Bahnhofstraße in Herne .
</item 10>
=== END ITEM id=3ee02783fd7486e4 ===

=== ITEM id=24cc52e0faa7f7bb ===
Gold sentences and facts:
Sentence 0: The Rhine -- Herne Canal ( German : Rhein - Herne - Kanal ) is a 45.6 - kilometre - long ( 28.3 mi ) transportation canal in the Ruhr area of North Rhine - Westphalia , Germany , with five canal locks . The canal was built over a period of eight years ( 5 April 1906 - 14 July 1914 ) and connects the harbour in Duisburg on the Rhine ( 51 ° 26 ′ 59 '' N 6 ° 46 ′ 1 '' E ﻿ / ﻿ 51.44972 ° N 6.76694 ° E ﻿ / 51.44972 ; 6.76694 ) with the Dortmund - Ems Canal near Henrichenburg ( 51 ° 37 ′ 1 '' N 7 ° 19 ′ 19 '' E ﻿ / ﻿ 51.61694 ° N 7.32194 ° E ﻿ / 51.61694 ; 7.32194 ) , following the valley of the Emscher . It was widened in the 1980s . The Rhein - Herne canal ship was designed specifically for this canal ; normally of about 1300 -- 1350 ton capacity , it has a maximum draft of 2.50 metres ( 8.2 ft ) , a length of approximately 80 metres ( 260 ft ) , and maximum beam of 9.50 metres ( 31.2 ft ) .
  Fact 0.0: The Rhine-Herne Canal (German: Rhein-Herne-Kanal) is a transportation canal.
  Fact 0.1: The Rhine-Herne Canal is 45.6 kilometres long (28.3 miles).
  Fact 0.2: The Rhine-Herne Canal is in the Ruhr area of North Rhine-Westphalia, Germany.
  Fact 0.3: The Rhine-Herne Canal has five canal locks.
  Fact 0.4: The canal was built over a period of eight years, from 5 April 1906 to 14 July 1914.
  Fact 0.5: The canal connects the harbour in Duisburg on the Rhine with the Dortmund-Ems Canal near Henrichenburg.
  Fact 0.6: The Duisburg harbour is at coordinates 51°26′59″N 6°46′1″E.
  Fact 0.7: The Henrichenburg end is at coordinates 51°37′1″N 7°19′19″E.
  Fact 0.8: The canal follows the valley of the Emscher.
  Fact 0.9: The canal was widened in the 1980s.
  Fact 0.10: The Rhein-Herne canal ship was designed specifically for this canal.
  Fact 0.11: The Rhein-Herne canal ship normally has a capacity of about 1300 to 1350 tons.
  Fact 0.12: The Rhein-Herne canal ship has a maximum draft of 2.50 metres (8.2 ft).
  Fact 0.13: The Rhein-Herne canal ship has a length of approximately 80 metres (260 ft).
  Fact 0.14: The Rhein-Herne canal ship has a maximum beam of 9.50 metres (31.2 ft).

Retrieved items:
<item 1>
Rhine–Herne Canal
The 6,900 - square - metre ( 1.7 - acre ) harbor basin was built from 1897 to 1898 when the section of the canal ( the then branch channel of the Dortmund - Ems
Canal ) , was already flooded . In 1912 , at the colliery port handled 360,096 tonnes of goods . In 1913 the docks expanded . Because of mining subsidence by
1954 the quaywall had to be increased by 2.50 metres ( 8.2 ft ) . In the 1980s , it was leased to Automobil - und Motorbootsport - Club Castrop - Rauxel e . V (
The car and motorboat club of Castrop - Rauxel ) and is used from recreational boating , being no longer used by Ruhrkohle AG , and contains many small boats .
The former mine railway , which starts at the harbour is now a walking and biking trail .
</item 1>

<item 2>
Rhine–Herne Canal
The industrial port is ~ 900 by 60 metres ( 2,950 by 200 ft ) and the trading port ~ 500 by 75 metres ( 1,640 by 246 ft ) . A private railway of length 18.4
kilometres ( 11.4 mi ) is operated which is linked to the tracks of the Deutsche Bahn Formerly the port dealt mainly with coal and mining products , now the
port is a major hub for petroleum products and second - largest inland port for cereals in North Rhine - Westphalia . It is operated by the Gelsenkirchener
Logistik - , Hafen - und Servicegesellschaft ( Gelsenkirchener logistics , harbour and service company . ) a subsidiary of Stadtwerke Gelsenkirchen . Port
Grimberg ( edit ) The port at Grimberg Hafen Grimberg ( Port Grimberg ) is on the south canal bank in Gelsenkirchen , near the city boundary to Herne . It was
named after the nearby Castle Grimberg , of which today only the foundations remain .
</item 2>

<item 3>
Rhine–Herne Canal
It was named after the nearby Castle Grimberg , of which today only the foundations remain . Built in 1914 by the Gelsenkirchener Bergwerks AG ( GBAG ) to
provide a link to the GBAG mines of Rheinelbe , Alma and Pluto as well as supplying the ironworks of ' Schalker group ' with imported ore . Around the same time
, and up to 1929 the industrial railway of Erzbahn was built , serving the Schalker and Bochum group iron and steelworks . Between 1937 and 1968 Grimberg
harbour was one of the major ports on the canal . In 1968 steel production at the southern end of the Erzbahn at Bochum ended , in 1982 the Schalker works
closed - with this the use of the port for transport of iron ore ended . Such was the emphasis on iron ore that much of the port now stands empty , part of it
nowadays is used for the processing of recycled materials for the building industry , as well as a storage facility .
</item 3>

<item 4>
Rhine–Herne Canal
From here maintenance and repair work are organised and performed . Oberhausen marina ( edit ) The Oberhausen marina is a yacht port opened in 2004 in close
proximity to the CentrO shopping centre . The port has a water depth of 2.5 metres ( 8.2 ft ) and over 13 bridges . Also at the Oberhausen marina are the
attractions Sealife Oberhausen and Oberhausen Modellbahn World . City port of Essen ( edit ) The Stadthafen Essen ( City port of Essen ) is the youngest port on
the Rhine -- Herne Canal opened in 1934 by the city of Essen , and operated since 1987 by Stadtwerke Essen .
</item 4>

<item 5>
Rhine–Herne Canal
The port consists of a doubled sided canal side quay together with a harbour port at right angles onto the canal , which together form a water area of 63,000
square metres ( 680,000 sq ft ) with associated dock - land of 300,000 square metres ( 3,200,000 sq ft ) The dock handles bulk goods such as solid mineral fuels
, petroleum and chemical products , stone , earth and iron and steel as well as recyclable materials , and is an important transhipment point . City port of
Gelsenkirchen ( edit ) The port of Gelsenkirchen The port of the city of Gelsenkirchen ( Stadthafen Gelsenkirchen ) was opened in July 1914 and is the oldest
port on the Rhine -- Herne Canal . The port has two docks branched from the canal situated just east of the lock at Gelsenkirchen and has a water surface of
117,800 square metres ( 1,268,000 sq ft ) .
</item 5>

<item 6>
Rhine–Herne Canal
The terminal is used as a mooring for pleasure craft . It forms part of the Waltrop lock park . Ecosystem ( edit ) Fish that inhabit the canal include European
eel , European carp , zander , carp bream , common roach and common rudd , as well as the less common pike and rainbow trout . Cities ( edit ) Duisburg
Oberhausen Essen Bottrop Gelsenkirchen Herne Recklinghausen Castrop - Rauxel Waltrop Datteln Wikimedia Commons has media related to Rhein - Herne - Kanal .
References and notes ( edit ) Jump up ^ Route A42 of the Federal Motorways now partly follows the route of the disused canal , ie . the section from the port at
the Friedrich der Große coal mine to Bahnhofstraße in Herne .
</item 6>

<item 7>
Rhine–Herne Canal
The canal was built over a period of eight years ( 5 April 1906 - 14 July 1914 ) and connects the harbour in Duisburg on the Rhine ( 51 ° 26 ′ 59 '' N 6 ° 46 ′
1 '' E ﻿ / ﻿ 51.44972 ° N 6.76694 ° E ﻿ / 51.44972 ; 6.76694 ) with the Dortmund - Ems Canal near Henrichenburg ( 51 ° 37 ′ 1 '' N 7 ° 19 ′ 19 '' E ﻿ / ﻿
51.61694 ° N 7.32194 ° E ﻿ / 51.61694 ; 7.32194 ) , following the valley of the Emscher . It was widened in the 1980s .
</item 7>

<item 8>
Rhine–Herne Canal
Jump up ^ Stadthafen Essen route-industriekultur.de Jump up ^ Hafen Gelsenkirchen hafen-ge.de Jump up ^ Stadthafen Gelsenkirchen route-industriekultur.de Jump
up ^ Hafen Grimberg und Erzbahn route-industrielkultur.de ^ Jump up to : Westhafen Wanne is more clearly on the north bank of the canal , with Osthafen Wanne
almost directly opposite on the south bank . Both are just east of the Schleuse Wanne - Eickel ( Wanne - Eickel lock ) Jump up ^ In German the port is named
Stadthafen Recklinghausen ^ Jump up to : Aerial image of Hafen der Zeche König Ludwig , ' Hafen der Zeche Friedrich der Große and Schleuse Herne - Ost
multimap.com ^ Jump up to : Aerial image of Hafen der Zeche König Ludwig , ' Hafen der Zeche Friedrich der Große and Schleuse Herne - Ost maps.google.co.uk Jump
up ^ Hafen Victor route-industriekultur.
</item 8>

<item 9>
Rhine–Herne Canal
Contents ( hide ) 1 Water system 2 Locks 2.1 Duisburg - Meiderich sluicegate 3 Ports 3.1 Duisburg 3.2 Oberhausen marina 3.3 City port of Essen 3.4 City port of
Gelsenkirchen 3.5 Port Grimberg 3.6 Ports Wanne west and Wanne east 3.7 City port of Recklinghausen 3.8 Port of the Frederick the Great mine 3.9 Port of the
King Ludwig mine 3.10 Port Victor 3.11 Port Rütgers 3.12 Port Luck 3.13 Port Marmorit 3.14 Waltrop lock park 4 Ecosystem 5 Cities 6 References and notes Water
system ( edit ) The distribution of water in the canal is realised through five pump stations . Located at Duisburg - Meiderich , Oberhausen , Gelsenkirchen ,
Wanne - Eickel and East Herne , they pump water from the Rhine into the canal at each stage .
</item 9>

<item 10>
Graph query row:
city: Gelsenkirchen
relationship: RELATED_TO
canal: Rhine–Herne Canal
</item 10>
=== END ITEM id=24cc52e0faa7f7bb ===

=== ITEM id=7bd2385d2b398642 ===
Gold sentences and facts:
Sentence 0: `` Knightfall '' is a 1993 - 1994 Batman story arc published by DC Comics . It consists of a trilogy of storylines that ran from 1993 to 1994 , consisting of `` Knightfall '' , `` Knightquest '' , and `` KnightsEnd '' . The story takes place over approximately six months . Bruce Wayne ( Batman ) suffers burnout and is systematically assaulted and crippled by a `` super steroid '' - enhanced genius named Bane . Wayne is replaced as Batman by an apprentice named Jean - Paul Valley ( a.k.a. Azrael ) , who becomes increasingly violent and unstable , tarnishing Batman 's reputation . Eventually , Wayne is healed through paranormal means , and reclaims his role as Batman .
  Fact 0.0: 'Knightfall' is a 1993-1994 Batman story arc published by DC Comics.
  Fact 0.1: Knightfall consists of a trilogy of storylines that ran from 1993 to 1994.
  Fact 0.2: The trilogy consists of 'Knightfall', 'Knightquest', and 'KnightsEnd'.
  Fact 0.3: The story takes place over approximately six months.
  Fact 0.4: Bruce Wayne (Batman) suffers burnout.
  Fact 0.5: Bruce Wayne (Batman) is systematically assaulted and crippled by a 'super steroid'-enhanced genius named Bane.
  Fact 0.6: Wayne is replaced as Batman by an apprentice named Jean-Paul Valley (a.k.a. Azrael).
  Fact 0.7: Jean-Paul Valley becomes increasingly violent and unstable.
  Fact 0.8: Jean-Paul Valley tarnishes Batman's reputation.
  Fact 0.9: Eventually, Wayne is healed through paranormal means.
  Fact 0.10: Eventually, Wayne reclaims his role as Batman.

Retrieved items:
<item 1>
Batman: Knightfall
When Bruce finally returns for good , he wears a sturdier , all - black Batsuit made of Kevlar ( inspired by the costume of the Tim Burton Batman films ) , and
drives a new , state - of - the - art Batmobile . He fights former Soviet agents Colonel Vega ( who teamed with Asp in `` Knightquest : The Search '' ) ,
KGBeast , and Dark Rider , in order to foil a plot to nuke Gotham City with a device the size and shape of a baseball . ( Troika is the Russian word for `` trio
'' . ) The saga also shows how Batman makes changes to his life as Bruce Wayne , his relationships with his `` family '' , plans to live without Alfred , and
copes with the decision of making Jean - Paul his replacement . Nightwing : Alfred 's return ( edit ) A one - shot that features the return of Alfred after his
resignation during `` KnightQuest : The Search '' . It has Nightwing going to England in order to track Alfred down . Azrael ( edit ) Main article : Azrael (
comic book ) A new series following Jean - Paul Valley was begun in April 1995 , titled Azrael .
</item 1>

<item 2>
Batman: Knightfall
Bruce then opens a hatch to the outside , which covered the very hole he fell into as a child , allowing sunlight to enter the night lenses in Jean - Paul 's
helmet . After being momentarily blinded , Jean - Paul removes his cowl , sees Bruce standing over him in the original Batman costume and concedes defeat ,
saying `` You are the Batman ... You 've always been the Batman ... and I am nothing ... '' Bruce comforts Jean - Paul , who leaves to wander the streets of
Gotham , homeless and destitute . Bruce decides not to take Jean - Paul to the police because it was his decision to make Jean - Paul the Batman , leading to
his subsequent breakdown . `` KnightsEnd '' was collected into a trade paperback about a year later . Originally released as Batman : KnightsEnd , recent
editions retitled it as Knightfall Volume 3 . Aftermath ( edit ) `` Prodigal '' ( edit ) Dick Grayson assuming the role of Batman during the `` Prodigal ''
storyline from Batman # 513 ( Dec. 1994 ) , a homage to Dick 's first appearance as Robin in Detective Comics # 38 ( 1940 ) . Art by Mike Gustovich .
</item 2>

<item 3>
Batman: Knightfall
During the breakout at Arkham Asylum , Batman and Robin both wear a black arm band with the S - shield engraved on it . The two stories involved DC Comics 's
most prominent characters . Similar stories followed for Green Lantern , Wonder Woman , Aquaman , and Green Arrow , with Green Lantern Hal Jordan being driven
insane and replaced with Kyle Rayner , a new Wonder Woman being appointed by Hippolyta , Aquaman losing his hand , and Green Arrow dying in a plane crash as his
long - lost son took over the role . In the opening of the novelisation , KnightFall , Dennis O'Neil stated that part of the reason `` Knightfall '' was written
was due to the recent popularity of more `` ruthless '' heroes such as the Terminator and James Bond in films , as editors were starting to wonder if readers
would prefer a Batman who was willing to kill his opponents . During DC vs. Marvel , Bane attempts to break Captain America 's back in a fight in a similar
maneuver , but is caught off - guard when Captain America 's shield returns to its owner to strike Bane in the back of the head and knock him out .
</item 3>

<item 4>
Batman: Knightfall
Their investigation leads them to the Caribbean and then Great Britain . Kinsolving 's brother - by - adoption Benedict Asp kidnapped her to use her special
powers to kill people at a distance . Asp demonstrates this new form of mass murder on a small English village . When Bruce Wayne finds Kinsolving , he finds
himself caught in the middle of a telekinetic tug - of - war between Asp and Kinsolving . The battle climaxes with her refocusing her energy to defeat Asp ; as
a side effect of the energy , Bruce 's broken spine becomes healed . However , the drugs forced onto her by Asp , combined with the effects of the fight with
Asp , reduce her mind to that of a child as Shondra 's traumatised mind regresses to the past to escape her unhappy present , and Wayne reluctantly puts her
into a mental institution . Bruce eventually leaves England to return home to a civilian life in Gotham , but Alfred remains in England , not wanting to see
Bruce Wayne damage his body further . He does not return to Gotham until a while later , when Dick Grayson persuades him to do so in later issues .
</item 4>

<item 5>
Batman: Knightfall
It consists of a trilogy of storylines that ran from 1993 to 1994 , consisting of `` Knightfall '' , `` Knightquest '' , and `` KnightsEnd '' . The story takes
place over approximately six months . Bruce Wayne ( Batman ) suffers burnout and is systematically assaulted and crippled by a `` super steroid '' - enhanced
genius named Bane . Wayne is replaced as Batman by an apprentice named Jean - Paul Valley ( a.k.a. Azrael ) , who becomes increasingly violent and unstable ,
tarnishing Batman 's reputation . Eventually , Wayne is healed through paranormal means , and reclaims his role as Batman . `` Knightfall '' resulted in long -
term ramifications for the Batman continuity , as Batman 's trust from the police , the public , and fellow superheroes had to be rebuilt , due to Azrael 's
violence . Additionally , Wayne realizes the peril and burden of attempting to work in solitude , leading to the eventual creation of the modern incarnation of
the Batman family . The events of Knightfall also led to the resignation of Wayne 's loyal butler , Alfred Pennyworth .
</item 5>

<item 6>
Batman: Knightfall
After this encounter , Bane 's men assault Batman before he himself makes his move and attacks him at Wayne Manor , his home as his alter - ego -- by this time
; Bane had deduced the secret identity of Batman . The fight between Bruce Wayne and Bane is detailed in Batman # 497 . By this time , Batman is so exhausted
from the exertion of catching the escaped criminals that he can not even defend himself ; Bane beats him mercilessly before breaking Wayne 's back over his knee
inside the Batcave below the manor , symbolically `` breaking '' Batman . Bane takes the grievously - wounded Wayne ( still costumed in the Batman outfit )
downtown to Gotham Square and throws him from a rooftop to demonstrate his superiority to the populace . Quick action by Robin and Alfred spares Batman 's life
, but at great cost - he is left a paraplegic . With Batman incapacitated , Bane assumes control of Gotham City 's underworld and takes over several illegal
operations within it . After his defeat , Bruce Wayne enlists the aid of Dr.
</item 6>

<item 7>
Batman: Knightfall
After his defeat , Bruce Wayne enlists the aid of Dr. Shondra Kinsolving to rehabilitate him and asks Jean - Paul Valley to take up the mantle of Batman so that
Gotham has a protector . Tim Drake argues with Bruce to allow Dick Grayson ( the former Robin and then - current Nightwing ) to become Batman , as he is more
experienced and mentally competent . Bruce replies that Grayson has his own responsibilities and would only take up the mantle of the Bat reluctantly ( Dick
later expressed resentment at not being asked to stand in as Batman ) . Bruce 's rationale for this decision is revealed in later issues - secretly , he does
not want Dick to have to face Bane , as he knows Dick 's character will compel him to try . Indeed , Bruce gives Jean - Paul strict orders never to engage Bane
in combat -- and when Jean - Paul does face Bane , only his modified gloves save Jean - Paul from being thrown to his death .
</item 7>

<item 8>
Batman: Knightfall
Art by Mike Gustovich . Bruce reaffirms his partnership with Tim , resolving the tension caused by Bruce 's unwillingness to accept help during the Arkham
prison break . Bruce passes the mantle of Batman to Grayson so he can re-evaluate what it will take to restore his aura of invincibility . This begins the
Prodigal storyline , a reference to Dick Grayson essentially being Bruce 's prodigal son ; Bruce had adopted Dick after his parents were murdered . Because of
the events of the entire arc , considerable time passes before Commissioner Gordon restores his trust in the idea of a Batman working for good . Gordon can tell
that he is not looking at the original Batman ( based on Jean - Paul 's costume and Dick 's height , and the fact that Jean - Paul was more than ready to kill
people ) , and he refuses to place blind trust in a costume after spending so long learning to trust the man .
</item 8>

<item 9>
Batman: Knightfall
In the climax to Year One of Injustice : Gods Among Us , Superman confronts Batman in the Batcave as Batman is downloading the formula to a pill that can grant
the user superhuman strength and durability , stolen from the Fortress of Solitude . While Superman does not want to kill Batman , he chooses to paralyze him by
breaking his back in a similar fashion to Bane ( albeit by standing on Batman 's back instead of slamming him over his knee as Bane did ) . This puts Batman out
of commission for much of Year Two as he heals . Reception ( edit ) IGN Comics # 15 ranked Batman : Knightfall Part One - Broken Bat on a list of the 25
greatest Batman graphic novels , saying : `` What makes ' Knightfall , Part One ' so memorable is not the actually snapping of Batman 's back . It 's the quick
fall into despair that proves most shocking . The ending is a foregone conclusion as Batman is worn down both physically and mentally . The Batman 's spirit is
broken before his vertebrae and that 's a feat you 'll never see accomplished anywhere else . ''
</item 9>

<item 10>
Batman: Knightfall
Returning to Gotham , Bruce meets with Tim . Even though Jean - Paul disobeyed Bruce 's order to refrain from attacking Bane , Bruce is sufficiently impressed
with Jean - Paul 's results . Bruce decides to retire and allow Jean - Paul to continue as Batman . But when Robin tells Bruce of the circumstances surrounding
Abattoir 's death , Bruce sneaks into the Batcave and demands that Jean - Paul step down . Jean - Paul refuses and tells Bruce to leave the cave and never come
back . To rehabilitate his skills due to his lost reflexes after so long out of action , Bruce asks the famed assassin Lady Shiva to retrain him ( using a mask
to conceal his identity and arguing that she will do it for the same reason that she does anything ; it might be interesting ) . Instead , Shiva pits Bruce
against several vengeful expert martial artists , having killed their master while wearing a distinctive tengu mask that carries a motif of a bat that she
subsequently gives to Batman . Shiva 's caveat is that these attacks will continue indefinitely until Bruce Wayne breaks his vow to never kill .
</item 10>
=== END ITEM id=7bd2385d2b398642 ===
