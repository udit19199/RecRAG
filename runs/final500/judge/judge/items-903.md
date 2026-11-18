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

=== ITEM id=4b57acbd95e87252 ===
Gold sentences and facts:
Sentence 0: Rank Team Won Lost Tied Pct . First NFL Season Total Games Division Dallas Cowboys 493 367 6 . 573 1960 866 NFC East Green Bay Packers 730 553 37 . 567 1921 1,320 NFC North Chicago Bears 744 568 42 . 565 1920 1,354 NFC North Miami Dolphins 439 341 . 563 1966 784 AFC East 5 New England Patriots 476 383 9 . 554 1960 868 AFC East 6 New York Giants 684 572 33 . 543 1925 1,289 NFC East 7 Denver Broncos 465 393 10 . 541 1960 868 AFC West 8 Minnesota Vikings 457 387 10 . 541 1961 854 NFC North 9 Baltimore Ravens 181 154 . 540 336 AFC North 10 San Francisco 49ers 522 450 14 . 537 1950 986 NFC West 11 Indianapolis Colts 502 441 7 . 532 1953 950 AFC South 12 Oakland Raiders 456 401 11 . 532 1960 868 AFC West 13 Pittsburgh Steelers 601 535 20 . 529 1933 1,156 AFC North 14 Kansas City Chiefs 447 409 12 . 522 1960 868 AFC West 15 Washington Redskins 586 572 28 . 506 1932 1,186 NFC East 16 Seattle Seahawks 325 318 . 505 1976 644 NFC West 17 Cleveland Browns 462 466 10 . 498 1950 938 AFC North 18 Los Angeles Chargers 426 431 11 . 497 1960 868 AFC West 19 Los Angeles Rams 544 554 21 . 496 1937 1,119 NFC West 20 Carolina Panthers 172 179 . 490 1995 352 NFC South 21 Philadelphia Eagles 555 591 26 . 485 1933 1,172 NFC East 22 Tennessee Titans 413 449 6 . 479 1960 868 AFC South 23 Buffalo Bills 400 460 8 . 465 1960 868 AFC East 24 Detroit Lions 544 641 32 . 460 1930 1,217 NFC North 25 Cincinnati Bengals 344 408 . 458 1968 756 AFC North 26 New York Jets 392 468 8 . 456 1960 868 AFC East 27 New Orleans Saints 338 427 5 . 442 1967 770 NFC South 28 Houston Texans 106 134 0 . 442 2002 240 AFC South 29 Jacksonville Jaguars 155 197 0 . 440 1995 352 AFC South 30 Atlanta Falcons 341 437 6 . 439 1966 784 NFC South 31 Arizona Cardinals 542 732 40 . 428 1920 1,314 NFC West 32 Tampa Bay Buccaneers 250 393 . 389 1976 644 NFC South
  Fact 0.0: The table columns are rank, team, won, lost, tied, pct, first NFL season, total games, and division.
  Fact 0.1: The Dallas Cowboys won 493 games, lost 367, tied 6, had a pct of .573, had a first NFL season of 1960, played 866 total games, and are in the NFC East.
  Fact 0.2: The Green Bay Packers won 730 games, lost 553, tied 37, had a pct of .567, had a first NFL season of 1921, played 1,320 total games, and are in the NFC North.
  Fact 0.3: The Chicago Bears won 744 games, lost 568, tied 42, had a pct of .565, had a first NFL season of 1920, played 1,354 total games, and are in the NFC North.
  Fact 0.4: The Miami Dolphins won 439 games, lost 341, had a pct of .563, had a first NFL season of 1966, played 784 total games, and are in the AFC East.
  Fact 0.5: Rank 5 is the New England Patriots, who won 476 games, lost 383, tied 9, had a pct of .554, had a first NFL season of 1960, played 868 total games, and are in the AFC East.
  Fact 0.6: Rank 6 is the New York Giants, who won 684 games, lost 572, tied 33, had a pct of .543, had a first NFL season of 1925, played 1,289 total games, and are in the NFC East.
  Fact 0.7: Rank 7 is the Denver Broncos, who won 465 games, lost 393, tied 10, had a pct of .541, had a first NFL season of 1960, played 868 total games, and are in the AFC West.
  Fact 0.8: Rank 8 is the Minnesota Vikings, who won 457 games, lost 387, tied 10, had a pct of .541, had a first NFL season of 1961, played 854 total games, and are in the NFC North.
  Fact 0.9: Rank 9 is the Baltimore Ravens, who won 181 games, lost 154, had a pct of .540, played 336 total games, and are in the AFC North.
  Fact 0.10: Rank 10 is the San Francisco 49ers, who won 522 games, lost 450, tied 14, had a pct of .537, had a first NFL season of 1950, played 986 total games, and are in the NFC West.
  Fact 0.11: Rank 11 is the Indianapolis Colts, who won 502 games, lost 441, tied 7, had a pct of .532, had a first NFL season of 1953, played 950 total games, and are in the AFC South.
  Fact 0.12: Rank 12 is the Oakland Raiders, who won 456 games, lost 401, tied 11, had a pct of .532, had a first NFL season of 1960, played 868 total games, and are in the AFC West.
  Fact 0.13: Rank 13 is the Pittsburgh Steelers, who won 601 games, lost 535, tied 20, had a pct of .529, had a first NFL season of 1933, played 1,156 total games, and are in the AFC North.
  Fact 0.14: Rank 14 is the Kansas City Chiefs, who won 447 games, lost 409, tied 12, had a pct of .522, had a first NFL season of 1960, played 868 total games, and are in the AFC West.
  Fact 0.15: Rank 15 is the Washington Redskins, who won 586 games, lost 572, tied 28, had a pct of .506, had a first NFL season of 1932, played 1,186 total games, and are in the NFC East.
  Fact 0.16: Rank 16 is the Seattle Seahawks, who won 325 games, lost 318, had a pct of .505, had a first NFL season of 1976, played 644 total games, and are in the NFC West.
  Fact 0.17: Rank 17 is the Cleveland Browns, who won 462 games, lost 466, tied 10, had a pct of .498, had a first NFL season of 1950, played 938 total games, and are in the AFC North.
  Fact 0.18: Rank 18 is the Los Angeles Chargers, who won 426 games, lost 431, tied 11, had a pct of .497, had a first NFL season of 1960, played 868 total games, and are in the AFC West.
  Fact 0.19: Rank 19 is the Los Angeles Rams, who won 544 games, lost 554, tied 21, had a pct of .496, had a first NFL season of 1937, played 1,119 total games, and are in the NFC West.
  Fact 0.20: Rank 20 is the Carolina Panthers, who won 172 games, lost 179, had a pct of .490, had a first NFL season of 1995, played 352 total games, and are in the NFC South.
  Fact 0.21: Rank 21 is the Philadelphia Eagles, who won 555 games, lost 591, tied 26, had a pct of .485, had a first NFL season of 1933, played 1,172 total games, and are in the NFC East.
  Fact 0.22: Rank 22 is the Tennessee Titans, who won 413 games, lost 449, tied 6, had a pct of .479, had a first NFL season of 1960, played 868 total games, and are in the AFC South.
  Fact 0.23: Rank 23 is the Buffalo Bills, who won 400 games, lost 460, tied 8, had a pct of .465, had a first NFL season of 1960, played 868 total games, and are in the AFC East.
  Fact 0.24: Rank 24 is the Detroit Lions, who won 544 games, lost 641, tied 32, had a pct of .460, had a first NFL season of 1930, played 1,217 total games, and are in the NFC North.
  Fact 0.25: Rank 25 is the Cincinnati Bengals, who won 344 games, lost 408, had a pct of .458, had a first NFL season of 1968, played 756 total games, and are in the AFC North.
  Fact 0.26: Rank 26 is the New York Jets, who won 392 games, lost 468, tied 8, had a pct of .456, had a first NFL season of 1960, played 868 total games, and are in the AFC East.
  Fact 0.27: Rank 27 is the New Orleans Saints, who won 338 games, lost 427, tied 5, had a pct of .442, had a first NFL season of 1967, played 770 total games, and are in the NFC South.
  Fact 0.28: Rank 28 is the Houston Texans, who won 106 games, lost 134, tied 0, had a pct of .442, had a first NFL season of 2002, played 240 total games, and are in the AFC South.
  Fact 0.29: Rank 29 is the Jacksonville Jaguars, who won 155 games, lost 197, tied 0, had a pct of .440, had a first NFL season of 1995, played 352 total games, and are in the AFC South.
  Fact 0.30: Rank 30 is the Atlanta Falcons, who won 341 games, lost 437, tied 6, had a pct of .439, had a first NFL season of 1966, played 784 total games, and are in the NFC South.
  Fact 0.31: Rank 31 is the Arizona Cardinals, who won 542 games, lost 732, tied 40, had a pct of .428, had a first NFL season of 1920, played 1,314 total games, and are in the NFC West.
  Fact 0.32: Rank 32 is the Tampa Bay Buccaneers, who won 250 games, lost 393, had a pct of .389, had a first NFL season of 1976, played 644 total games, and are in the NFC South.

Retrieved items:
<item 1>
NFL win–loss records
NFL win -- loss records - wikipedia NFL win -- loss records Jump to : navigation , search It has been suggested that NFL standings since AFL -- NFL merger be
merged into this article . ( Discuss ) Proposed since October 2017 . The following is a list of win -- loss records for each of the 32 active National Football
League ( NFL ) teams . Contents ( hide ) 1 Regular season 2 Playoffs 3 Notes 4 References Regular season ( edit ) George Halas ( right ) of the Chicago Bears ,
who have recorded the most regular season wins . The following is a listing of all 32 current National Football League ( NFL ) teams ranked by their regular
season win -- loss record percentage , accurate as of the end of the 2016 NFL regular season . For teams that previously played in the All - America Football
Conference ( AAFC ) , their AAFC win - loss records are not included as the NFL does not officially count AAFC statistics despite the 1950 NFL -- AAFC merger .
However , for teams that played in the American Football League ( AFL ) , their AFL win - loss records are included and officially counted by the NFL .
</item 1>

<item 2>
NFL win–loss records
540 336 AFC North 10 San Francisco 49ers 522 450 14 . 537 1950 986 NFC West 11 Indianapolis Colts 502 441 7 . 532 1953 950 AFC South 12 Oakland Raiders 456 401
11 . 532 1960 868 AFC West 13 Pittsburgh Steelers 601 535 20 . 529 1933 1,156 AFC North 14 Kansas City Chiefs 447 409 12 . 522 1960 868 AFC West 15 Washington
Redskins 586 572 28 . 506 1932 1,186 NFC East 16 Seattle Seahawks 325 318 . 505 1976 644 NFC West 17 Cleveland Browns 462 466 10 . 498 1950 938 AFC North 18 Los
Angeles Chargers 426 431 11 . 497 1960 868 AFC West 19 Los Angeles Rams 544 554 21 . 496 1937 1,119 NFC West 20 Carolina Panthers 172 179 . 490 1995 352 NFC
South 21 Philadelphia Eagles 555 591 26 .
</item 2>

<item 3>
List of National Football League records (individual)
Most career road losses , post-season , by a starting quarterback : 7 , Brett Favre , 1993 -- 2009 Most career home losses , post-season , by a starting
quarterback : 6 , Peyton Manning , 1999 -- 2016 . Most consecutive losses , post-season , by a starting quarterback : 4 , Y.A. Tittle , 1957 , 1961 -- 1963 ;
Warren Moon 1991 -- 1994 ; Andy Dalton 2011 -- 2014 Most consecutive road losses , post-season , by a starting quarterback : 6 , Dave Krieg , 1983 , 1984 , 1987
, 1992 -- 1994 Most consecutive road losses by a starting quarterback : 14 , Steve DeBerg , 1978 -- 1979 Most consecutive home losses by a starting quarterback
: 10 , Archie Manning , 1979 -- 1980 ; Chris Weinke , 2001 -- 2002 , 2006 Most road losses in a season , by a rookie starting quarterback : 8 , Peyton Manning ,
Indianapolis Colts 1998 ; Derek Carr , Oakland Raiders ,
</item 3>

<item 4>
NFL win–loss records
490 1995 352 NFC South 21 Philadelphia Eagles 555 591 26 . 485 1933 1,172 NFC East 22 Tennessee Titans 413 449 6 . 479 1960 868 AFC South 23 Buffalo Bills 400
460 8 . 465 1960 868 AFC East 24 Detroit Lions 544 641 32 . 460 1930 1,217 NFC North 25 Cincinnati Bengals 344 408 . 458 1968 756 AFC North 26 New York Jets 392
468 8 . 456 1960 868 AFC East 27 New Orleans Saints 338 427 5 . 442 1967 770 NFC South 28 Houston Texans 106 134 0 . 442 2002 240 AFC South 29 Jacksonville
Jaguars 155 197 0 . 440 1995 352 AFC South 30 Atlanta Falcons 341 437 6 . 439 1966 784 NFC South 31 Arizona Cardinals 542 732 40 . 428 1920 1,314 NFC West 32
Tampa Bay Buccaneers 250 393 .
</item 4>

<item 5>
NFL win–loss records
Ties are registered as a half win and a half loss when calculating the win - loss record . Best win -- loss record in division Rank Team Won Lost Tied Pct .
First NFL Season Total Games Division Dallas Cowboys 493 367 6 . 573 1960 866 NFC East Green Bay Packers 730 553 37 . 567 1921 1,320 NFC North Chicago Bears 744
568 42 . 565 1920 1,354 NFC North Miami Dolphins 439 341 . 563 1966 784 AFC East 5 New England Patriots 476 383 9 . 554 1960 868 AFC East 6 New York Giants 684
572 33 . 543 1925 1,289 NFC East 7 Denver Broncos 465 393 10 . 541 1960 868 AFC West 8 Minnesota Vikings 457 387 10 . 541 1961 854 NFC North 9 Baltimore Ravens
181 154 . 540 336 AFC North 10 San Francisco 49ers 522 450 14 .
</item 5>

<item 6>
NFL win–loss records
480 AFC East 18 Philadelphia Eagles 40 19 21 . 475 2013 NFC East 19 Jacksonville Jaguars 11 5 6 . 455 2007 AFC South 20 Los Angeles Rams 43 19 24 . 442 NFC West
21 ( tie ) Arizona Cardinals 16 7 9 . 438 2015 NFC West 21 ( tie ) New Orleans Saints 16 7 9 . 438 2013 NFC South 23 Houston Texans 7 . 429 2016 AFC South 24
Tennessee Titans 33 14 19 . 424 2008 AFC South 25 Atlanta Falcons 22 9 13 . 409 2016 NFC South 26 Minnesota Vikings 47 19 28 . 404 2015 NFC North 27 Tampa Bay
Buccaneers 15 6 9 . 400 2007 NFC South 28 Los Angeles Chargers 28 11 17 . 393 2013 AFC West 29 Cleveland Browns 31 11 20 . 355 2002 AFC North 30 Detroit Lions
20 7 13 .
</item 6>

<item 7>
NFL win–loss records
389 1976 644 NFC South Playoffs ( edit ) Members of the Baltimore Ravens franchise , the team with the highest playoff win -- loss percentage , during the
Lombardi Trophy presentation following their Super Bowl XLVII victory Marvin Lewis , head coach of the Cincinnati Bengals , who have registered the lowest
playoff win - loss percentage of all time The following is a listing of all 32 current National Football League ( NFL ) teams ranked by their playoff win - loss
record percentage , accurate as of the end of Super Bowl LI . Best win -- loss record in division Team active in playoffs Rank Team Games Won Lost Pct . Last
playoff appearance Division Baltimore Ravens 23 15 8 . 652 2014 AFC North New England Patriots 51 32 19 . 627 2016 AFC East Green Bay Packers 56 34 22 . 607
2016 NFC North 4 ( tie ) Pittsburgh Steelers 60 36 24 . 600 2016 AFC North 4 ( tie ) San Francisco 49ers 50 30 20 . 600 2013 NFC West 6 Oakland Raiders 44 25 19
.
</item 7>

<item 8>
NFL win–loss records
355 2002 AFC North 30 Detroit Lions 20 7 13 . 350 2016 NFC North 31 Kansas City Chiefs 26 9 17 . 346 2016 AFC West 32 Cincinnati Bengals 19 5 14 . 263 2015 AFC
North Notes ( edit ) Jump up ^ For example , the Browns ' regular season win -- loss record does not include a 47 -- 4 -- 3 record compiled in the All - America
Football Conference from 1946 to 1949 , nor does their playoff record include the 5 -- 0 record compiled during the same period . Likewise , the 49ers ' AAFC
regular season win -- loss record of 38 -- 14 -- 2 , nor their 1 - 1 AAFC playoff record are officially counted Jump up ^ The Patriots played as the `` Boston
Patriots '' from 1960 -- 1970 , before adopting the `` New England Patriots '' name , which they have played under since the 1971 season . The team was also
briefly known as the `` Bay State Patriots '' prior to the 1971 season .
</item 8>

<item 9>
List of National Football League records (individual)
Peyton Manning , Indianapolis Colts 1998 ; Derek Carr , Oakland Raiders , 2014 Most losses in a season , by a starting quarterback : 15 , Archie Manning , New
Orleans Saints , 1980 ; Jeff George , Indianapolis Colts , 1991 ; Deshone Kizer , Cleveland Browns , 2017 Most consecutive losses , by a rookie starting
quarterback : 15 , Deshone Kizer , Cleveland Browns , 2017 Most losses in a season , by a rookie starting quarterback : 15 , Deshone Kizer , Cleveland Browns ,
2017 Most consecutive losses in a season , by a starting quarterback : 15 , Deshone Kizer , Cleveland Browns , 2017 Most consecutive losses to start a season ,
by a starting quarterback : 15 , Deshone Kizer , Cleveland Browns , 2017 Most consecutive losses by a starting quarterback : 21 , Dan Pastorini , Houston Oilers
, 1972 -- 1974 Most consecutive losses by a starting quarterback , regular season , to start a career : 15 , Deshone Kizer , Cleveland Browns , 2017 Most
consecutive road losses by a starting quarterback , regular season , to start a career : 13 ,
</item 9>

<item 10>
NFL win–loss records
600 2013 NFC West 6 Oakland Raiders 44 25 19 . 568 2016 AFC West 7 Carolina Panthers 16 9 7 . 563 2015 NFC South 8 Dallas Cowboys 61 34 27 . 557 2016 NFC East 9
( tie ) Washington Redskins 42 23 19 . 548 2015 NFC East 9 ( tie ) Denver Broncos 42 23 19 . 548 2015 AFC West 11 Seattle Seahawks 31 16 15 . 516 2016 NFC West
12 New York Giants 49 24 25 . 490 2016 NFC East 13 Indianapolis Colts 45 22 23 . 489 2014 AFC South 14 Miami Dolphins 41 20 21 . 488 2016 AFC East 15 Chicago
Bears 35 17 18 . 486 NFC North 16 Buffalo Bills 29 14 15 . 483 1999 AFC East 17 New York Jets 25 12 13 . 480 AFC East 18 Philadelphia Eagles 40 19 21 .
</item 10>
=== END ITEM id=4b57acbd95e87252 ===

=== ITEM id=c59b07ac224f33a8 ===
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
London Summer Olympics 2012 US $ 10,400,000,000 US $ 14,600,000,000 US $ 4,400,000,000 GBP £ nil 2012 Additional costs include $90 million for converting the
Olympic Stadium ( London ) to a football venue Sochi Winter Olympics 2014 US $ 51,000,000,000 US $ 53,150,000 The most expensive Olympic Games in history ,
surpassing the previous record set by the 2008 Beijing Summer Olympic Games . Rio de Janeiro Summer Olympics 2016 US $ 13,100,000,000 US $ 11.6 billion
Pyeongchang Winter Olympics 2018 US $ 12,900,000,000 Tokyo Summer Olympics 2020 Beijing Winter Olympics 2022 Paris Summer Olympics 2024 TBA 2026 Winter Olympics
2026 Los Angeles Summer Olympics 2028 TBA 2032 See also ( edit ) List of megaprojects References ( edit ) Jump up ^ Bennetts , Marc ( 19 January 2014 ) .
</item 1>

<item 2>
Cost of the Olympic Games
International Olympic Committee . Retrieved 2014 - 02 - 13 . ^ Jump up to : `` Official Report of the XXII Olympiad Moscow 1980 '' ( PDF ) . International
Olympic Committee . 1981 . Retrieved 2014 - 02 - 13 . ^ Jump up to : `` FINAL REPORT published by the Organising Committee of the XlVth Winter Olympic Games
1984 at Sarajevo '' ( PDF ) . ^ Jump up to : `` Official Report of the Games of the XXIIIrd Olympiad Los Angeles , 1984 '' ( PDF ) . International Olympic
Committee . 1985 . Retrieved 2014 - 02 - 13 . ^ Jump up to : Julia Campbell . `` Cost to Host Olympic Games Skyrockets '' . ABC News . Retrieved August 6 , 2013
. ^ Jump up to : `` Olympic Cities : Booms and Busts '' . Cnbc.com. 2012 - 01 - 19 . Retrieved 2014 - 02 - 07 .
</item 2>

<item 3>
Cost of the Olympic Games
Jump up ^ `` Olympic Cities : Booms and Busts '' . Cnbc.com. 2012 - 01 - 19 . Retrieved 2014 - 02 - 07 . Jump up ^ `` Turin Winter Olympics 2006 - 2014 - 02 -
06 - CIPRA : Living in the Alps '' . Retrieved 2014 - 02 - 12 . Jump up ^ `` Olympic Cities : Booms and Busts '' . Cnbc.com. 2012 - 01 - 19 . Retrieved 2014 -
02 - 07 . Jump up ^ `` Beijing Olympics to cost China 44,000,000,000 dollars - 2008 - 06 - 08 - Pravda News '' . Retrieved 2014 - 02 - 12 . Jump up ^ `` Beijing
Olympics ' profit exceeds 1b yuan '' . Jump up ^ `` Olympics cost B.C. $925 M '' . CBC Sports . July 9 , 2010 . Retrieved February 4 , 2013 .
</item 3>

<item 4>
Cost of the Olympic Games
July 9 , 2010 . Retrieved February 14 , 2013 . Jump up ^ `` London 2012 - Final cost of London 2012 Games revealed '' . Yahoo . October 23 , 2012 . Retrieved
April 3 , 2013 . Jump up ^ Gibson , Owen ( October 24 , 2012 ) . `` Olympic Games 2012 olympics olys , Paralympics 2012 , Olympic legacy , Olympic Stadium ,
London ( News ) , UK news , Politics , Sport '' . The Guardian . London . Jump up ^ `` London Olympics exceed initial budget by £ 6.52 bn '' . Newstatesman.com
. Retrieved 2014 - 02 - 07 . Jump up ^ `` Locog London Organising Committee of the Olympic and Paralympic Games , Olympic Games 2012 olympics olys , British
Olympic Association , Sport politics , Sport '' ( PDF ) . LOCOG FINAL ANNUAL REPORT Page 33 . London . March 31 , 2012 .
</item 4>

<item 5>
Cost of the Olympic Games
Cost of the Olympic Games - wikipedia Cost of the Olympic Games Jump to : navigation , search This article possibly contains original research . Please improve
it by verifying the claims made and adding inline citations . Statements consisting only of original research should be removed . ( February 2018 ) ( Learn how
and when to remove this template message ) Sports - related costs for the Summer Games since 1960 is on average USD 5.2 billion and for the Winter Games USD 3.1
billion dollars . Severe cost overruns made the 2014 Winter Olympics the most expensive Olympics in history ; with Russian politician Boris Nemtsov citing
allegations of corruption among government officials , and Allison Stewart of the Saïd Business School at Oxford citing tight relationships between the
government and construction firms . While originally budgeted at US $12 billion , the budget expanded to US $51 billion , surpassing the estimated $44 billion
cost of the 2008 Summer Olympics in Beijing , making Sochi the most expensive Olympics in history . The table below is listing the not so hard costs of the
Games .
</item 5>

<item 6>
Cost of the Olympic Games
Beijing Summer Olympics 2008 US $ 44,000,000,000 CNY 1,000,000,000 ( US $ 146,000,000 ) 2008 Vancouver Winter Olympics CDN $ 1,700,000,000 ( US $ 1,260,000,000
) US $ 6,400,000,000 US $ 2,300,000,000 CDN $ 1,900,000 Included in the total US $ 6,400,000,000 cost are the $1,000,000,000 for security , $2,500,000,000 for
transportation extensions and upgrades , and $900,000,000 for the new Vancouver Convention Centre ( An additional $554,000,000 was spent by the city including a
portion on the Olympic Village ) .
</item 6>

<item 7>
Cost of the Olympic Games
2002 - 09 - 18 . Retrieved 2010 - 02 - 13 . Jump up ^ `` As Olympic glow fades , Athens Questions $15,000,000,000 Cost - 2008 - 07 - 21 - The Christian Science
Monitor '' . Retrieved 2010 - 02 - 13 . Jump up ^ `` How the 2004 Olympics Triggered Greece 's Decline - 2012 - 08 - 02 - Bloomberg Business '' . Retrieved 2014
- 02 - 12 . Jump up ^ `` Olympics ' may cost Greece dear ' - 2004 - 06 - 02 - BBC NEWS '' . BBC News . 2004 - 06 - 02 . Retrieved 2014 - 02 - 12 . Jump up ^
Schlotterbeck , Bianca ( 19 January 2012 ) . `` Athens , Greece ( Summer , 2004 ) '' . Olympic Cities : Booms and Busts . CNBC . Retrieved 13 February 2013 .
Jump up ^ `` Olympic Cities : Booms and Busts '' .
</item 7>

<item 8>
Cost of the Olympic Games
Retrieved 2014 - 02 - 07 . ^ Jump up to : Nina Berglund . `` Lillehammer marks OL anniversary '' . News In English . Retrieved February 7 , 2014 . compared to
the NOK 8,500,000,000 spent on Lillehammer ^ Jump up to : `` Exchange rate , US dollar ( USD ) '' . Norges Bank . Retrieved February 7 , 2014 . January 1994
7.5109 ^ Jump up to : Schlotterbeck , Bianca ( 19 January 2012 ) . `` Nagano , Japan ( Winter 1998 ) '' . Olympic Cities : Booms and Busts . CNBC . Retrieved 13
February 2013 . Jump up ^ Craig , David ( 2009 ) . Squandered . Constable & Robinson Limited . p. 300 . ISBN 9781849011617 . Jump up ^ `` Olympic glory at any
price ? - John Madden & James Giesecke - 26 Jul 2012 , Business Spectator '' .
</item 8>

<item 9>
Cost of the Olympic Games
Retrieved 2014 - 02 - 07 . ^ Jump up to : `` Beijing Olympiad : Profit or Loss ? '' . China Internet Information Center . Retrieved 2008 - 08 - 17 . ^ Jump up
to : `` XV Olympic Winter Games : Official Report '' ( PDF ) . International Olympic Committee . 1988 . Retrieved 2014 - 02 - 13 . Jump up ^ Flyvbjerg , Bent ;
Allison Stewart ( 2012 ) . `` Olympic Proportions : Cost and Cost Overrun at the Olympics 1960 - 2012 '' . Working Paper . Saïd Business School , University of
Oxford . Jump up ^ Brunet , Ferran ( 1995 ) . An economic analysis of the Barcelona'92 Olympic Games : resources , financing and impacts `` . Working Paper .
Universitat Autònoma de Barcelona ^ Jump up to : `` Olympic Cities : Booms and Busts '' . Cnbc.com. 2012 - 01 - 19 . Retrieved 2014 - 02 - 07 .
</item 9>

<item 10>
Cost of the Olympic Games
Moritz Winter Olympics 1928 CHF 706,000 CHF104 , 800 Los Angeles Summer Olympics 1932 $1,000,000 Berlin Summer Olympics 1936 $30,000,000 London Summer Olympics
1948 £ 761,688 £ 29,000 Helsinki Summer Olympics 1952 1,580,000,000 mk 49,000,000 mk Melbourne Summer Olympics 1956 A £ 5,400,000 A £ 300,000 Tokyo Summer
Olympics 1964 US $ 72,000,000 Mexico City Summer Olympics 1968 US $ 176,000,000 Munich Summer Olympics 1972 1,972,000,000 DM Montreal Summer Olympics 1976 CDN $
207,000,000 CDN $ 1,410,000,000 CDN $ 990,000,000 2006 A special tobacco tax was introduced in May 1976 to fund the loss Lake Placid Winter Olympics 1980 US $
49,000,000 US $ 169,000,000 Moscow Summer Olympics 1980 US $ 231,000,000 US $ 1,350,000,000 USD $1,
</item 10>
=== END ITEM id=c59b07ac224f33a8 ===

=== ITEM id=26be186988c566fe ===
Gold sentences and facts:
Sentence 0: The Great Gatsby is a 1925 novel written by American author F. Scott Fitzgerald that follows a cast of characters living in the fictional town of West Egg on prosperous Long Island in the summer of 1922 . The story primarily concerns the young and mysterious millionaire Jay Gatsby and his quixotic passion and obsession for the beautiful former debutante Daisy Buchanan . Considered to be Fitzgerald 's magnum opus , The Great Gatsby explores themes of decadence , idealism , resistance to change , social upheaval , and excess , creating a portrait of the Jazz Age or the Roaring Twenties that has been described as a cautionary tale regarding the American Dream .
  Fact 0.0: The Great Gatsby is a 1925 novel.
  Fact 0.1: The Great Gatsby was written by American author F. Scott Fitzgerald.
  Fact 0.2: The Great Gatsby follows a cast of characters living in the fictional town of West Egg on prosperous Long Island in the summer of 1922.
  Fact 0.3: The fictional town of West Egg is on prosperous Long Island.
  Fact 0.4: The story of The Great Gatsby primarily concerns the young and mysterious millionaire Jay Gatsby.
  Fact 0.5: The story of The Great Gatsby concerns Jay Gatsby's quixotic passion and obsession for the beautiful former debutante Daisy Buchanan.
  Fact 0.6: Daisy Buchanan is a beautiful former debutante.
  Fact 0.7: The Great Gatsby is considered to be Fitzgerald's magnum opus.
  Fact 0.8: The Great Gatsby explores themes of decadence, idealism, resistance to change, social upheaval, and excess.
  Fact 0.9: The Great Gatsby creates a portrait of the Jazz Age, also called the Roaring Twenties.
  Fact 0.10: The Great Gatsby has been described as a cautionary tale regarding the American Dream.
Sentence 1: Set on the prosperous Long Island of 1922 , The Great Gatsby provides a critical social history of America during the Roaring Twenties within its fictional narrative . That era , known for unprecedented economic prosperity , the evolution of jazz music flapper culture , new technologies in communication ( motion pictures , broadcast radio , recorded music ) forging a genuine mass culture ; and bootlegging , along with other criminal activity , is plausibly depicted in Fitzgerald 's novel . Fitzgerald uses many of these societal developments of the 1920s that were to build Gatsby 's stories from many of the simple details like automobiles to broader themes like Fitzgerald 's discreet allusions to the organized crime culture which was the source of Gatsby 's fortune . Fitzgerald depicts the garish society of the Roaring Twenties by placing the book 's plotline within the historical context of the era .
  Fact 1.0: The Great Gatsby is set on the prosperous Long Island of 1922.
  Fact 1.1: The Great Gatsby provides a critical social history of America during the Roaring Twenties within its fictional narrative.
  Fact 1.2: The Roaring Twenties era was known for unprecedented economic prosperity.
  Fact 1.3: The Roaring Twenties era was known for the evolution of jazz music and flapper culture.
  Fact 1.4: The Roaring Twenties era saw new technologies in communication: motion pictures, broadcast radio, and recorded music.
  Fact 1.5: The new communication technologies of the Roaring Twenties forged a genuine mass culture.
  Fact 1.6: The Roaring Twenties era saw bootlegging, along with other criminal activity.
  Fact 1.7: Bootlegging and other criminal activity of the Roaring Twenties are plausibly depicted in Fitzgerald's novel.
  Fact 1.8: Fitzgerald uses many societal developments of the 1920s to build the stories of Gatsby.
  Fact 1.9: Fitzgerald uses these societal developments in his novel, from simple details like automobiles to broader themes.
  Fact 1.10: Fitzgerald makes discreet allusions in his novel to the organized crime culture.
  Fact 1.11: The organized crime culture was the source of Gatsby's fortune.
  Fact 1.12: Fitzgerald depicts the garish society of the Roaring Twenties.
  Fact 1.13: Fitzgerald depicts the garish society by placing the book's plotline within the historical context of the era.

Retrieved items:
<item 1>
The Great Gatsby
They introduce Nick to Jordan Baker , an attractive , cynical young golfer with whom Nick begins a romantic relationship . She reveals to Nick that Tom has a
mistress , Myrtle Wilson , who lives in the `` valley of ashes '' , an industrial dumping ground between West Egg and New York City . Not long after this
revelation , Nick travels to New York City with Tom and Myrtle to an apartment Tom keeps for his affairs with Myrtle and others . At Tom 's New York apartment ,
a vulgar and bizarre party takes place . It ends with Tom breaking Myrtle 's nose after she annoys him by saying Daisy 's name several times . The Plaza Hotel
in the early - 1920s As the summer progresses , Nick eventually receives an invitation to one of Gatsby 's parties . Nick encounters Jordan Baker at the party ,
and they meet Gatsby himself , an aloof and surprisingly young man who recognizes Nick from their same division in the Great War . Through Jordan , Nick later
learns that Gatsby knew Daisy through a purely chance meeting in 1917 , when Daisy and her friends were doing volunteer services ' work with young Officers
headed to Europe .
</item 1>

<item 2>
The Great Gatsby
The story deals with the limits and realities of America 's myths of social and class mobility ; and the inevitably hopeless lower class aspirations to rise
above the station ( s ) of their birth . The book in stark relief through the narrator , Nick Carraway , observes that : `` ... a sense of the fundamental
decencies is parcelled out unequally at birth . '' Using elements of irony and tragic ending , it also delves into themes of excesses of the rich , and
recklessness of youth . Others , like journalist Nick Gillespie , see The Great Gatsby as a story `` underlying permanence of class differences ; even in the
face of a modern economy 's attempt to assert that the class structure is based ; not on status and inherited position ; but , upon the innovation and the
ability of literally anyone , to succeed by meeting the ever - changing demands and tastes of consumers ' needs . ''
</item 2>

<item 3>
The Great Gatsby
Fitzgerald uses many of these societal developments of the 1920s that were to build Gatsby 's stories from many of the simple details like automobiles to
broader themes like Fitzgerald 's discreet allusions to the organized crime culture which was the source of Gatsby 's fortune . Fitzgerald depicts the garish
society of the Roaring Twenties by placing the book 's plotline within the historical context of the era . Fitzgerald 's visits to Long Island 's north shore
and his experience attending parties at mansions inspired The Great Gatsby 's setting . Today , there are a number of theories as to which mansion was the
inspiration for the book . One possibility is Land 's End , a notable Gold Coast Mansion where Fitzgerald may have attended a party . Many of the events in
Fitzgerald 's early life are reflected throughout The Great Gatsby . Fitzgerald was a young man from Minnesota , and like Nick , he was educated at an Ivy
League school , Princeton ( in Nick 's case , Yale ) .
</item 3>

<item 4>
The Great Gatsby
The only other titles that seem to fit it are Trimalchio and On the Road to West Egg . I had two others , Gold - hatted Gatsby and The High - bouncing Lover ,
but they seemed too slight . ' Jump up ^ Wulick , Dr. Anna . `` Understanding The Great Gatsby First Line and Epigraph '' . Retrieved 2016 - 12 - 15 . Jump up ^
F. Scott Fitzgerald , Chapter 7 opening sentence , The Great Gatsby Jump up ^ Tanner 's introduction to the Penguin edition ( 2000 ) , p. vii -- viii . Jump up
^ Hill , W. Speed ; Burns , Edward M. ; Shillingsburg , Peter L. ( 2002 ) . Text : An Interdisciplinary Annual of Textual Studies . 14 . University of Michigan
Press . ISBN 0472112724 ... , because in early November he wrote Perkins that `` I have now decided to stick to the title I put on the book . Trimalchio in West
Egg . Jump up ^ Fitzgerald , Francis Scott ; Perkins , Maxwell ( 1971 ) .
</item 4>

<item 5>
The Great Gatsby
Nick stages an unsettlingly small funeral for Gatsby in which none of Gatsby 's associates or partygoers attend . Later , Nick runs into Tom in New York and
finds out that Tom had told George that Gatsby was Myrtle 's secret lover and that Gatsby had killed her , then gave Gatsby 's address to George . Nick breaks
up with Jordan , and , disillusioned with the East , moves back to the Midwest . Major characters ( edit ) Nick Carraway -- a Yale graduate originating from the
Midwest , a World War I veteran , and , at the start of the plot , a newly arrived resident of West Egg , who is aged 29 ( later 30 ) . He also serves as the
first - person narrator of the novel . He is Gatsby 's next - door neighbor and a bond salesman . He is easy - going , occasionally sarcastic , and somewhat
optimistic , although this latter quality fades as the novel progresses . He is a more grounded character than the others , and more practical , and is always
in awe of other character 's lifestyles and morals .
</item 5>

<item 6>
The Great Gatsby
In this novel , Great Neck ( King 's Point ) became the `` new money '' peninsula of West Egg and Port Washington ( Sands Point ) the old - money East Egg .
Several mansions in the area served as inspiration for Gatsby 's home , such as Oheka Castle and Beacon Towers , since demolished . By mid-1923 , Fitzgerald had
written 18,000 words for his novel but discarded most of his new story as a false start , some of which resurfaced in the 1924 short story `` Absolution '' .
Work on The Great Gatsby began in earnest in April 1924 . Fitzgerald wrote in his ledger , `` Out of woods at last and starting novel . '' He decided to make a
departure from the writing process of his previous novels and told Perkins that the novel was to be a `` consciously artistic achievement '' and a `` purely
creative work -- not trashy imaginings as in my stories but the sustained imagination of a sincere and yet radiant world . '' He added later , during editing ,
that he felt `` an enormous power in me now , more than I 've ever had . ''
</item 6>

<item 7>
The Great Gatsby
The play failed miserably , and Fitzgerald worked that winter on magazine stories struggling to pay his debt caused by the production . The stories were , in
his words , `` all trash and it nearly broke my heart , '' although included among those stories was `` Winter Dreams '' , which Fitzgerald later described as
`` a sort of first draft of the Gatsby idea '' . After the birth of their child , the Fitzgeralds moved to Great Neck , New York , on Long Island , in October
1922 . The town was used as the scene of The Great Gatsby . Fitzgerald 's neighbors in Great Neck included such prominent and newly wealthy New Yorkers as
writer Ring Lardner , actor Lew Fields , and comedian Ed Wynn . These figures were all considered to be `` new money '' , unlike those who came from Manhasset
Neck or Cow Neck Peninsula , places which were home to many of New York 's wealthiest established families , and which sat across the bay from Great Neck . This
real - life juxtaposition gave Fitzgerald his idea for `` West Egg '' and `` East Egg '' .
</item 7>

<item 8>
The Great Gatsby
In her book Careless People : Murder , Mayhem and the Invention of ' The Great Gatsby ( 2013 ) , Sarah Churchwell speculates that parts of the ending of The
Great Gatsby were based on the Hall - Mills Case . Based on her forensic search for clues , she asserts that the two victims in the Hall - Mills murder case
inspired the characters who were murdered in The Great Gatsby . Plot summary ( edit ) The main events of the novel take place in the summer of 1922 . Nick
Carraway , a Yale graduate and veteran of the Great War from the Midwest -- who serves as the novel 's narrator -- takes a job in New York as a bond salesman .
He rents a small house on Long Island , in the fictional village of West Egg , next door to the lavish mansion of Jay Gatsby , a mysterious multi-millionaire
who holds extravagant parties but does not participate in them . Nick drives around the bay to East Egg for dinner at the home of his cousin , Daisy Fay
Buchanan , and her husband , Tom , a college acquaintance of Nick 's .
</item 8>

<item 9>
The Great Gatsby
Although this passage has some resemblance to the painting , a closer explanation can be found in the description of Daisy Buchanan as the `` girl whose
disembodied face floated along the dark cornices and blinding signs . '' Ernest Hemingway wrote in A Moveable Feast that when Fitzgerald lent him a copy of The
Great Gatsby to read , he immediately disliked the cover , but `` Scott told me not to be put off by it , that it had to do with a billboard along a highway in
Long Island that was important in the story . He said he had liked the jacket and now he did n't like it . '' Title ( edit ) Fitzgerald had difficulty choosing
a title for his novel and entertained many choices before reluctantly choosing The Great Gatsby , a title inspired by Alain - Fournier 's Le Grand Meaulnes .
Prior , Fitzgerald shifted between Gatsby ; Among Ash - Heaps and Millionaires ; Trimalchio ; Trimalchio in West Egg ; On the Road to West Egg ; Under the Red ,
White , and Blue ; Gold - Hatted Gatsby ; and The High - Bouncing Lover .
</item 9>

<item 10>
The Great Gatsby
The Great Gatsby - Wikipedia The Great Gatsby Jump to : navigation , search This article is about the novel . For the film , TV and opera adaptations , see The
Great Gatsby ( disambiguation ) . The Great Gatsby Cover of the first edition in 1925 . Author F. Scott Fitzgerald Cover artist Francis Cugat Country United
States Language English Genre Novel Published April 10 , 1925 ( Charles Scribner 's Sons ) Media type Print ( hardcover & paperback ) Preceded by The Beautiful
and Damned ( 1922 ) Followed by Tender Is the Night ( 1934 ) The Great Gatsby is a 1925 novel written by American author F. Scott Fitzgerald that follows a cast
of characters living in the fictional town of West Egg on prosperous Long Island in the summer of 1922 . The story primarily concerns the young and mysterious
millionaire Jay Gatsby and his quixotic passion and obsession for the beautiful former debutante Daisy Buchanan .
</item 10>
=== END ITEM id=26be186988c566fe ===

=== ITEM id=d31d46809ae65ba6 ===
Gold sentences and facts:
Sentence 0: `` It Ai n't Me '' is a song by Norwegian DJ Kygo and American singer Selena Gomez . It was released by Interscope Records , Sony and Ultra on 17 February 2017 as the lead single from Kygo 's first EP , Stargazing . The song was written by Kygo , Gomez , Andrew Watt , Brian Lee and Ali Tamposi . It was produced by Kygo , Watt , Ben Rice and Louis Bell . A dance - pop , electropop and tropical house song , `` It Ai n't Me '' comprises an acoustic guitar line , and a build - drop arrangement in its chorus featuring pulsing piano notes , bass , synthesizers , finger - snap claps and pan flute melodies . Gomez sings the track in a husky tone , while in the chorus her vocals are reduced to recurring syllables . The lyrics are nostalgic and narrate a past relationship ruined by alcoholism and partying too often .
  Fact 0.0: 'It Ai n't Me' is a song by Norwegian DJ Kygo and American singer Selena Gomez.
  Fact 0.1: 'It Ai n't Me' was released by Interscope Records, Sony and Ultra on 17 February 2017.
  Fact 0.2: 'It Ai n't Me' was the lead single from Kygo's first EP, Stargazing.
  Fact 0.3: 'It Ai n't Me' was written by Kygo, Selena Gomez, Andrew Watt, Brian Lee and Ali Tamposi.
  Fact 0.4: 'It Ai n't Me' was produced by Kygo, Andrew Watt, Ben Rice and Louis Bell.
  Fact 0.5: 'It Ai n't Me' is a dance-pop, electropop and tropical house song.
  Fact 0.6: 'It Ai n't Me' comprises an acoustic guitar line.
  Fact 0.7: 'It Ai n't Me' has a build-drop arrangement in its chorus.
  Fact 0.8: The chorus of 'It Ai n't Me' features pulsing piano notes, bass, synthesizers, finger-snap claps and pan flute melodies.
  Fact 0.9: Selena Gomez sings the track in a husky tone.
  Fact 0.10: In the chorus of 'It Ai n't Me', Selena Gomez's vocals are reduced to recurring syllables.
  Fact 0.11: The lyrics of 'It Ai n't Me' are nostalgic.
  Fact 0.12: The lyrics of 'It Ai n't Me' narrate a past relationship ruined by alcoholism and partying too often.

Retrieved items:
<item 1>
It Ain't Me
Enter It Ai n't Me in the search field and then press Enter . Jump up ^ `` American single certifications -- Kygo & Selena Gomez -- It Ai n't Me '' . Recording
Industry Association of America . Retrieved 7 June 2017 . If necessary , click Advanced , then click Format , then select Single , then click SEARCH Jump up ^
`` Kygo & Selena Gomez - It Ai n't Me RadioDate '' . Radioairplay.fm. 16 February 2017 . Retrieved 9 September 2017 . Jump up ^ ( 1 ) Jump up ^ `` BBC - Radio 1
- Playlist '' . BBC Radio 1 . 17 March 2017 . Archived from the original on 18 March 2017 . Retrieved 21 June 2017 . Jump up ^ `` Music Meeting for March 28 ''
. Play IGA . Archived from the original on 28 March 2017 . External links ( edit ) Lyrics of this song at MetroLyrics Kygo Discography Studio albums Cloud Nine
( 2016 ) Kids in Love ( 2017 ) EPs Stargazing ( 2017
</item 1>

<item 2>
It Ain't Me
According to Tamposi , the producers brought the song to life . The three songwriters would usually spend several months pitching their songs to different
singers , but with `` It Ai n't Me '' Gomez heard it about a month afterward and recorded it a few days later . Tamposi was called back for the vocal sessions
with Gomez where they shared a similar emotional vision for `` It Ai n't Me '' , basing it on their own personal experiences . Recalling the sessions , Tamposi
said she was impressed with Gomez 's involvement and the ideas she came up with for the song . Music and lyrics ( edit ) A dance - pop , electropop and tropical
house song , `` It Ai n't Me '' is composed in the key of C Major and is set in the time signature of alla breve at a tempo of 100 beats per minute . Gomez 's
vocal range spawns from the low note G to the high note E , while the music follows the chord progression of Am -- C -- F -- C -- F -- C -- G. While the piano
later in the song plays in a Chord Progression of : Am -- C -- G -- F -- Am -- C -- G .
</item 2>

<item 3>
It Ain't Me
It was released by Interscope Records , Sony and Ultra on 17 February 2017 as the lead single from Kygo 's first EP , Stargazing . The song was written by Kygo
, Gomez , Andrew Watt , Brian Lee and Ali Tamposi . It was produced by Kygo , Watt , Ben Rice and Louis Bell . A dance - pop , electropop and tropical house
song , `` It Ai n't Me '' comprises an acoustic guitar line , and a build - drop arrangement in its chorus featuring pulsing piano notes , bass , synthesizers ,
finger - snap claps and pan flute melodies . Gomez sings the track in a husky tone , while in the chorus her vocals are reduced to recurring syllables . The
lyrics are nostalgic and narrate a past relationship ruined by alcoholism and partying too often . In addition to reaching number one in Croatia , Lebanon , and
Norway , the song attained a top five peak in Australia , Austria , Belgium , Canada , the Czech Republic , Denmark , Finland , France , Germany , Greece ,
Hungary , Ireland , the Netherlands , New Zealand , Poland , Portugal , Slovakia , Sweden and Switzerland .
</item 3>

<item 4>
It Ain't Me
For the Bob Dylan song , see It Ai n't Me , Babe . `` It Ai n't Me '' Single by Kygo and Selena Gomez from the EP Stargazing Released 17 February 2017 Format
Digital download Recorded 2016 Genre Dance - pop electropop tropical house Length 3 : 40 Label Interscope Sony Ultra Songwriter ( s ) Kyrre Gørvell - Dahll
Selena Gomez Andrew Wotman Brian Lee Ali Tamposi Producer ( s ) Kygo Andrew Watt Ben Rice Louis Bell Kygo singles chronology `` Carry Me '' ( 2016 ) `` It Ai
n't Me '' ( 2017 ) `` First Time '' ( 2017 ) Selena Gomez singles chronology `` Trust Nobody '' ( 2016 ) Trust Nobody 2016 `` It Ai n't Me '' ( 2017 ) It Ai n't
Me2017 `` Bad Liar '' ( 2017 ) Bad Liar 2017 Music video `` It Ai n't Me '' on YouTube `` It Ai n't Me '' is a song by Norwegian DJ Kygo and American singer
Selena Gomez .
</item 4>

<item 5>
It Ain't Me
The same synthesizers from the verses are repeated with a piano in the background . Gomez 's vocals in the drop are reduced to syllables from her verses such as
a recurring `` whiskey neat '' and are accompanied by percussion instruments . `` It Ai n't Me '' is a breakup song with a theme of nostalgia . The lyrics are
about standing up for yourself in the face of a toxic relationship . Gomez narrates the regret of a previous relationship ruined by her former lover 's habits
of drinking and partying too often . The lyrics specifically address alcohol abuse , a type of alcoholism . Tamposi explained that `` It Ai n't Me '' is about a
woman finding the strength to walk away from a toxic relationship despite the pressure she receives from society to stand by her man . Although the lyrics are
sombre , the song has a distinctly uplifting and empowering sound . Release ( edit ) In early February 2017 , `` It Ai n't Me '' was registered on the American
Society of Composers , Authors and Publishers ( ASCAP ) database . Gomez first teased the single with two Instagram stories on 3 February 2017 .
</item 5>

<item 6>
It Ain't Me
It also reached the top 10 in Italy , Spain , the United Kingdom and the United States . Contents ( hide ) 1 Writing and recording 2 Music and lyrics 3 Release
4 Critical reception 5 Chart performance 6 Music video 7 Usage in media 8 Track listing 9 Credits and personnel 10 Charts 10.1 Weekly charts 10.2 Year - end
charts 11 Certifications 12 Release history 13 References 14 External links Writing and Recording ( edit ) The lyrics of `` It Ai n't Me '' were written by
Andrew Watt , Brian Lee and Ali Tamposi . With Watt previously working with Kygo , the DJ 's manager Myles Shear contacted Watt for a session with Kygo which
Watt also facilitated for Lee and Tamposi . During their session for `` It Ai n't Me '' , they initially found the track very poor , but were trying to salvage
it . Kygo and Shear suggested that they continue working on the song .
</item 6>

<item 7>
It Ain't Me
The song carries a calm tone and a drop common in tropical house music , but is complimented by an equal measure of electronic and pop rock elements . The track
starts with a folk - like acoustic guitar riff . It is followed immediately by Gomez singing the first verse in a husky tone . As the song develops , a steady
bass and synthesizers are heard . The production is subtle and varied with finger snaps sometimes being louder than the kick drum , and the guitar riff and
piano notes alternating in and out . In the chorus , Gomez is accompanied by background vocals that pertain to the sound of a choir . `` It Ai n't Me '' A 21 -
second sound sample from `` It Ai n't Me '' featuring the chorus to the start of the drop which Gomez sings accompanied by backing vocals , piano notes , a
guitar riff , and vocal effects . Problems playing this file ? See media help . The chorus has a new melodic arrangement and a crescendo . It is ensued by chord
- buoyed builds and drops comprising finger snaps and pan flute melodies . The same synthesizers from the verses are repeated with a piano in the background .
</item 7>

<item 8>
It Ain't Me
Usage in Media ( edit ) The song is featured in tenth episode of the television series The Bold Type . Track listing ( edit ) Digital download `` It Ai n't Me
'' -- 3 : 40 Digital download -- Tiësto 's AFTR : HRS Remix `` It Ai n't Me '' ( Tiësto 's AFTR : HRS Remix ) -- 3 : 12 Credits and personnel ( edit ) Credits
adapted from Qobuz . Kygo -- composer , producer Selena Gomez -- composer , vocals Andrew Watt -- guitar , producer , vocal producer , composer , songwriter
Brian Lee -- composer , songwriter Ali Tamposi -- composer , songwriter Tom Coyne -- mastering engineer Serban Ghenea -- mixing engineer Ben Rice -- vocal
producer Louis Bell -- vocal producer Charts ( edit ) Weekly charts ( edit ) Chart ( 2017 ) Peak position Australia ( ARIA ) Austria ( Ö3 Austria Top 40 )
Belgium ( Ultratop 50 Flanders ) Belgium ( Ultratop 50 Wallonia ) Canada ( Canadian Hot 100 ) Canada CHR / Top 40 ( Billboard ) Canada Hot AC ( Billboard ) 6
Colombia ( Monitor Latino Anglo ) Croatia ( HRT ) Czech Republic (
</item 8>

<item 9>
It Ain't Me
It Ai n't Me - wikipedia It Ai n't Me Jump to : navigation , search This article is about the Kygo and Selena Gomez song . For the Bob Dylan song , see It Ai
n't Me , Babe .
</item 9>

<item 10>
It Ain't Me
000 United States ( RIAA ) Platinum 1,000,000 sales figures based on certification alone shipments figures based on certification alone sales + streaming
figures based on certification alone Release history ( edit ) Region Date Version Format Label Ref . Italy 16 February 2017 Original Contemporary hit radio Sony
Various 17 February 2017 Single Digital download Sony Ultra United States 28 February 2017 Original Mainstream radio Ultra 7 March 2017 Rhythmic radio Sony
Ultra United Kingdom 17 March 2017 Contemporary hit radio United States 28 March 2017 Hot adult contemporary Various 7 April 2017 Tiësto 's AFTR : HRS Remix
Digital download References ( edit ) Jump up ^ https://www.pandora.com/artist/kygo-and-selena-gomez/it-aint-me-single/it-aint-me/TR9c9kdKhgc9Xcm ^ Jump up to :
Bein , Kat ( 23 February 2017 ) . `` Selena Gomez Is ' an Old Soul With a Beautiful Baby Face , ' Says ' It Ai n't Me ' Songwriter '' . Billboard . Retrieved 24
February 2016 .
</item 10>
=== END ITEM id=d31d46809ae65ba6 ===

=== ITEM id=a3bd4f68e24eddde ===
Gold sentences and facts:
Sentence 0: In the light - dependent reactions ( mainly three vital reaction steps : ( I ) Hill and Bendall 's Z - scheme of cyclic and non-cyclic reactions , ( II ) Mitchell 's chemi - osmosis for ATP synthesis and ( III ) Dolai 's S - state diagrams in water - splitting ) , one molecule of the pigment chlorophyll absorbs one photon and loses one electron . This electron is passed to a modified form of chlorophyll called pheophytin , which passes the electron to a quinone molecule , starting the flow of electrons down an electron transport chain that leads to the ultimate reduction of NADP to NADPH . In addition , this creates a proton gradient ( energy gradient ) across the chloroplast membrane , which is used by ATP synthase in the synthesis of ATP . The chlorophyll molecule ultimately regains the electron it lost when a water molecule is split in a process called photolysis , which releases a dioxygen ( O ) molecule as a waste product .
  Fact 0.0: The light-dependent reactions have three vital reaction steps.
  Fact 0.1: Step I of the light-dependent reactions is Hill and Bendall's Z-scheme of cyclic and non-cyclic reactions.
  Fact 0.2: Step II of the light-dependent reactions is Mitchell's chemi-osmosis for ATP synthesis.
  Fact 0.3: Step III of the light-dependent reactions is Dolai's S-state diagrams in water-splitting.
  Fact 0.4: In the light-dependent reactions, one molecule of the pigment chlorophyll absorbs one photon.
  Fact 0.5: In the light-dependent reactions, one molecule of chlorophyll loses one electron.
  Fact 0.6: The lost electron is passed to a modified form of chlorophyll called pheophytin.
  Fact 0.7: Pheophytin passes the electron to a quinone molecule.
  Fact 0.8: This starts the flow of electrons down an electron transport chain.
  Fact 0.9: The electron transport chain leads to the ultimate reduction of NADP to NADPH.
  Fact 0.10: The electron flow creates a proton gradient (energy gradient) across the chloroplast membrane.
  Fact 0.11: ATP synthase uses the proton gradient in the synthesis of ATP.
  Fact 0.12: The chlorophyll molecule ultimately regains the electron it lost.
  Fact 0.13: The electron is regained when a water molecule is split in a process called photolysis.
  Fact 0.14: Photolysis releases a dioxygen (O) molecule as a waste product.
Sentence 1: Although photosynthesis is performed differently by different species , the process always begins when energy from light is absorbed by proteins called reaction centres that contain green chlorophyll pigments . In plants , these proteins are held inside organelles called chloroplasts , which are most abundant in leaf cells , while in bacteria they are embedded in the plasma membrane . In these light - dependent reactions , some energy is used to strip electrons from suitable substances , such as water , producing oxygen gas . The hydrogen freed by the splitting of water is used in the creation of two further compounds that act as an immediate energy storage means : reduced nicotinamide adenine dinucleotide phosphate ( NADPH ) and adenosine triphosphate ( ATP ) , the `` energy currency '' of cells .
  Fact 1.0: Photosynthesis is performed differently by different species.
  Fact 1.1: Photosynthesis always begins when energy from light is absorbed by proteins called reaction centres.
  Fact 1.2: Reaction centres are proteins that contain green chlorophyll pigments.
  Fact 1.3: In plants, reaction centres are held inside organelles called chloroplasts.
  Fact 1.4: Chloroplasts are most abundant in leaf cells.
  Fact 1.5: In bacteria, reaction centres are embedded in the plasma membrane.
  Fact 1.6: In the light-dependent reactions, some energy is used to strip electrons from suitable substances, such as water.
  Fact 1.7: The light-dependent reactions produce oxygen gas.
  Fact 1.8: The hydrogen freed by splitting water is used in the creation of two further compounds.
  Fact 1.9: The two compounds act as an immediate energy storage means.
  Fact 1.10: The two compounds are reduced nicotinamide adenine dinucleotide phosphate (NADPH) and adenosine triphosphate (ATP).
  Fact 1.11: ATP is the energy currency of cells.
Sentence 2: Photosynthesis occurs in two stages . In the first stage , light - dependent reactions or light reactions capture the energy of light and use it to make the energy - storage molecules ATP and NADPH . During the second stage , the light - independent reactions use these products to capture and reduce carbon dioxide .
  Fact 2.0: Photosynthesis occurs in two stages.
  Fact 2.1: In the first stage, light-dependent reactions (light reactions) capture the energy of light.
  Fact 2.2: In the first stage, the light-dependent reactions use the captured energy to make the energy-storage molecules ATP and NADPH.
  Fact 2.3: During the second stage, the light-independent reactions use ATP and NADPH.
  Fact 2.4: In the second stage, the light-independent reactions capture and reduce carbon dioxide.

Retrieved items:
<item 1>
Photosynthesis
Louis N.M. Duysens and Jan Amesz discovered that chlorophyll a will absorb one light , oxidize cytochrome f , chlorophyll a ( and other pigments ) will absorb
another light , but will reduce this same oxidized cytochrome , stating the two light reactions are in series . Development of the concept In 1893 , Charles
Reid Barnes proposed two terms , photosyntax and photosynthesis , for the biological process of synthesis of complex carbon compounds out of carbonic acid , in
the presence of chlorophyll , under the influence of light . Over time , the term photosynthesis came into common usage as the term of choice . Later discovery
of anoxygenic photosynthetic bacteria and photophosphorylation necessitated redefinition of the term . C3 : C4 photosynthesis Research After WWII at late 1940
at the University of California , Berkeley , the details of photosynthetic carbon metabolism were sorted out by the chemists Melvin Calvin , Andrew Benson ,
James Bassham and a score of students and researchers utilizing the carbon - 14 isotope and paper chromatography techniques .
</item 1>

<item 2>
Photosynthesis
Cornelis Van Niel made key discoveries explaining the chemistry of photosynthesis . By studying purple sulfur bacteria and green bacteria he was the first to
demonstrate that photosynthesis is a light - dependent redox reaction , in which hydrogen reduces carbon dioxide . Robert Emerson discovered two light reactions
by testing plant productivity using different wavelengths of light . With the red alone , the light reactions were suppressed . When blue and red were combined
, the output was much more substantial . Thus , there were two photosystems , one absorbing up to 600 nm wavelengths , the other up to 700 nm . The former is
known as PSII , the latter is PSI . PSI contains only chlorophyll `` a '' , PSII contains primarily chlorophyll `` a '' with most of the available chlorophyll
`` b '' , among other pigment . These include phycobilins , which are the red and blue pigments of red and blue algae respectively , and fucoxanthol for brown
algae and diatoms .
</item 2>

<item 3>
Photosynthesis
The process is most productive when the absorption of quanta are equal in both the PSII and PSI , assuring that input energy from the antenna complex is divided
between the PSI and PSII system , which in turn powers the photochemistry . Melvin Calvin works in his photosynthesis laboratory . Robert Hill thought that a
complex of reactions consisting of an intermediate to cytochrome b ( now a plastoquinone ) , another is from cytochrome f to a step in the carbohydrate -
generating mechanisms . These are linked by plastoquinone , which does require energy to reduce cytochrome f for it is a sufficient reductant . Further
experiments to prove that the oxygen developed during the photosynthesis of green plants came from water , were performed by Hill in 1937 and 1939 . He showed
that isolated chloroplasts give off oxygen in the presence of unnatural reducing agents like iron oxalate , ferricyanide or benzoquinone after exposure to light
. The Hill reaction is as follows : 2 H O + 2 A + ( light , chloroplasts ) → 2 AH + O where A is the electron acceptor .
</item 3>

<item 4>
Photosynthesis
The overall equation for the light - independent reactions in green plants is 3 CO + 9 ATP + 6 NADPH + 6 H → C H O - phosphate + 9 ADP + 8 P + 6 NADP + 3 H O
Overview of the Calvin cycle and carbon fixation Carbon fixation produces the intermediate three - carbon sugar product , which is then converted to the final
carbohydrate products . The simple carbon sugars produced by photosynthesis are then used in the forming of other organic compounds , such as the building
material cellulose , the precursors for lipid and amino acid biosynthesis , or as a fuel in cellular respiration . The latter occurs not only in plants but also
in animals when the energy from plants is passed through a food chain . The fixation or reduction of carbon dioxide is a process in which carbon dioxide
combines with a five - carbon sugar , ribulose 1 , 5 - bisphosphate , to yield two molecules of a three - carbon compound , glycerate 3 - phosphate , also known
as 3 - phosphoglycerate .
</item 4>

<item 5>
Photosynthesis
but canceling n water molecules from each side gives the net equation : CO + H O + photons → ( CH O ) + O carbon dioxide + water + light energy → carbohydrate +
oxygen Other processes substitute other compounds ( such as arsenite ) for water in the electron - supply role ; for example some microbes use sunlight to
oxidize arsenite to arsenate : The equation for this reaction is : CO + ( AsO 3 − 3 ) + photons → ( AsO 3 − 4 ) + CO carbon dioxide + arsenite + light energy →
arsenate + carbon monoxide ( used to build other compounds in subsequent reactions ) Photosynthesis occurs in two stages . In the first stage , light -
dependent reactions or light reactions capture the energy of light and use it to make the energy - storage molecules ATP and NADPH . During the second stage ,
the light - independent reactions use these products to capture and reduce carbon dioxide . Most organisms that utilize oxygenic photosynthesis use visible
light for the light - dependent reactions , although at least three use shortwave infrared or , more specifically , far - red radiation . Some organisms employ
even more radical variants of photosynthesis .
</item 5>

<item 6>
Photosynthesis
Contents ( hide ) 1 Overview 2 Photosynthetic membranes and organelles 3 Light - dependent reactions 3.1 Z scheme 3.2 Water photolysis 4 Light - independent
reactions 4.1 Calvin cycle 4.2 Carbon concentrating mechanisms 4.2. 1 On land 4.2. 2 In water 5 Order and kinetics 6 Efficiency 7 Evolution 7.1 Symbiosis and
the origin of chloroplasts 7.2 Cyanobacteria and the evolution of photosynthesis 8 Discovery 8.1 Development of the concept 8.2 C3 : C4 photosynthesis research
9 Factors 9.1 Light intensity ( irradiance ) , wavelength and temperature 9.2 Carbon dioxide levels and photorespiration 10 See also 11 References 12 Further
reading 12.1 Books 12.2 Papers 13 External links Overview Photosynthesis changes sunlight into chemical energy , splits water to liberate O , and fixes CO into
sugar . Photosynthetic organisms are photoautotrophs , which means that they are able to synthesize food directly from carbon dioxide and water using energy
from light .
</item 6>

<item 7>
Photosynthesis
Z scheme The `` Z scheme '' In plants , light - dependent reactions occur in the thylakoid membranes of the chloroplasts where they drive the synthesis of ATP
and NADPH . The light - dependent reactions are of two forms : cyclic and non-cyclic . In the non-cyclic reaction , the photons are captured in the light -
harvesting antenna complexes of photosystem II by chlorophyll and other accessory pigments ( see diagram at right ) . The absorption of a photon by the antenna
complex frees an electron by a process called photoinduced charge separation . The antenna system is at the core of the chlorophyll molecule of the photosystem
II reaction center . That freed electron is transferred to the primary electron - acceptor molecule , pheophytin . As the electrons are shuttled through an
electron transport chain ( the so - called Z - scheme shown in the diagram ) , it initially functions to generate a chemiosmotic potential by pumping proton
cations ( H ) across the membrane and into the thylakoid space .
</item 7>

<item 8>
Photosynthesis
The cyclic reaction takes place only at photosystem I. Once the electron is displaced from the photosystem , the electron is passed down the electron acceptor
molecules and returns to photosystem I , from where it was emitted , hence the name cyclic reaction . Water photolysis Main articles : Photodissociation and
Oxygen evolution The NADPH is the main reducing agent produced by chloroplasts , which then goes on to provide a source of energetic electrons in other cellular
reactions . Its production leaves chlorophyll in photosystem I with a deficit of electrons ( chlorophyll has been oxidized ) , which must be balanced by some
other reducing agent that will supply the missing electron . The excited electrons lost from chlorophyll from photosystem I are supplied from the electron
transport chain by plastocyanin . However , since photosystem II is the first step of the Z - scheme , an external source of electrons is required to reduce its
oxidized chlorophyll a molecules . The source of electrons in green - plant and cyanobacterial photosynthesis is water .
</item 8>

<item 9>
Photosynthesis
IOSR Journal of Pharmacy and Biological Sciences . 12 ( 6 ) : 65 - 67 . doi : 10.9790 / 3008 - 1206026567 . ISSN 2319 - 7676 . Photosystem II is the only known
biological enzyme that carries out this oxidation of water . The hydrogen ions released contribute to the transmembrane chemiosmotic potential that leads to ATP
synthesis . Oxygen is a waste product of light - dependent reactions , but the majority of organisms on Earth use oxygen for cellular respiration , including
photosynthetic organisms . Light - independent reactions Calvin cycle Main articles : Calvin cycle , Carbon fixation , and Light - independent reactions In the
light - independent ( or `` dark '' ) reactions , the enzyme RuBisCO captures CO from the atmosphere and , in a process called the Calvin - Benson cycle , it
uses the newly formed NADPH and releases three - carbon sugars , which are later combined to form sucrose and starch .
</item 9>

<item 10>
Photosynthesis
Light - dependent reactions Light - dependent reactions of photosynthesis at the thylakoid membrane Main article : Light - dependent reactions In the light -
dependent reactions ( mainly three vital reaction steps : ( I ) Hill and Bendall 's Z - scheme of cyclic and non-cyclic reactions , ( II ) Mitchell 's chemi -
osmosis for ATP synthesis and ( III ) Dolai 's S - state diagrams in water - splitting ) , one molecule of the pigment chlorophyll absorbs one photon and loses
one electron . This electron is passed to a modified form of chlorophyll called pheophytin , which passes the electron to a quinone molecule , starting the flow
of electrons down an electron transport chain that leads to the ultimate reduction of NADP to NADPH . In addition , this creates a proton gradient ( energy
gradient ) across the chloroplast membrane , which is used by ATP synthase in the synthesis of ATP . The chlorophyll molecule ultimately regains the electron it
lost when a water molecule is split in a process called photolysis , which releases a dioxygen ( O ) molecule as a waste product .
</item 10>
=== END ITEM id=a3bd4f68e24eddde ===

=== ITEM id=3efa1d9220e5d0ee ===
Gold sentences and facts:
Sentence 0: Symphony No. 40 in G minor , KV. 550 was written by Wolfgang Amadeus Mozart in 1788 . It is sometimes referred to as the `` Great G minor symphony '' , to distinguish it from the `` Little G minor symphony '' , No. 25 . The two are the only extant minor key symphonies Mozart wrote .
  Fact 0.0: Symphony No. 40 in G minor, KV. 550 was written by Wolfgang Amadeus Mozart.
  Fact 0.1: Symphony No. 40 in G minor, KV. 550 was written in 1788.
  Fact 0.2: Symphony No. 40 in G minor, KV. 550 is sometimes referred to as the 'Great G minor symphony'.
  Fact 0.3: The Great G minor symphony is distinguished from the 'Little G minor symphony', No. 25.
  Fact 0.4: Symphony No. 40 and the Little G minor symphony, No. 25, are the only extant minor key symphonies Mozart wrote.

Retrieved items:
<item 1>
Symphony No. 40 (Mozart)
Jump up ^ Milada Jonášová : `` Eine Aufführung der g - moll - Sinfonie KV 550 bei Baron van Swieten im Beisein Mozarts '' , in : Mozart Studien 20 , Tutzing
2011 , pp. 253 -- 268 . An abridged English translation was published in the Newsletter of the Mozart Society of America 16.1 ( 2012 ) , currently available on
line at ( 2 ) . The letter reads `` im Wien habe ich selbst es von verstorbenem Mozart gehört , als Er sie bei Baron Wanswiten ( sic ) hat produciren lassen ,
das ( s ) er wärend der production aus dem Zimmer sich hat entfernen müssen , wie man Sie unrichtig aufgeführt hat '' , meaning `` and in Vienna I have heard
myself from the departed Mozart , that when he had it performed in Baron Wanswiten 's rooms , he had to leave the room during the performance because it was
being played so incorrectly '' . Jump up ^ List from Zaslaw 1983 .
</item 1>

<item 2>
Symphony No. 40 (Mozart)
The contrapuntal opening bars of this movement appear thus in keyboard reduction : The minuet begins with an angry , cross-accented hemiola rhythm and a pair of
three - bar phrases , as shown in the following piano reduction : Various commentators have asserted that while the music is labelled `` minuet '' , it would
hardly be suitable for dancing . The contrasting gentle trio section , in G major , alternates the playing of the string section with that of the winds . The
fourth movement opens with a series of rapidly ascending notes outlining the tonic triad illustrating what is commonly referred to as the Mannheim rocket . The
movement is written largely in eight - bar phrases , following the general tendency toward rhythmic squareness in the finales of classical - era symphonies . A
remarkable modulating passage in which every tone in the chromatic scale but one is played , strongly destabilising the key , occurs at the beginning of the
development section . The single note left out is in fact a G ( the tonic ) . The symphony typically has a duration of about 25 minutes . Reception ( edit )
This work has elicited varying interpretations from critics .
</item 2>

<item 3>
Symphony No. 40 (Mozart)
334 . Jump up ^ Hopkins 1981 . Jump up ^ Nottebohm 1887 , p. 531 . Jump up ^ Zaslaw and Cowdery 1990 , p. 210 . Jump up ^ See No. 38 , the aria `` Erblicke hier
, bethörter Mensch '' . Source : Zaslaw and Cowdery 1990 , p. 210 ) . Jump up ^ `` Wolfgang Amadeus Mozart - Symphony in G Minor '' . Discography of American
Historical Recordings . References ( edit ) Biba , Otto ( 2009 ) . `` Kommentar '' , Wolfgang Amadeus Mozart . Sinfonie g - moll , KV 550 . Autographe Partitur
. Erste und zweite Fassung . Faksimile - Ausgabe . Gesellschaft der Musikfreunde in Wien , p. 8 . Deutsch , Otto Erich ( 1965 ) . Mozart : A Documentary
Biography . Stanford , CA : Stanford University Press .
</item 3>

<item 4>
Symphony No. 40 (Mozart)
40 The apartment where Mozart wrote his last three Symphonies : Michael Lorenz , `` Mozart 's Apartment on the Alsergrund '' Article online Wolfgang Amadeus
Mozart Biography Biographies Birthplace Grand tour Name Nationality Scatology Smallpox Italy Berlin Prague Death Music Köchel catalogue List of compositions
Concert arias , songs , canons Dances Horn concertos Masses Operas Piano concertos Works for solo piano Sonatas Symphonies Compositional method Family Leopold
Mozart Anna Maria Mozart Maria Anna Mozart ( Nannerl ) Constanze Mozart Maria Anna Thekla Mozart ( Bäsle ) Franz Xaver Wolfgang Mozart Karl Thomas Mozart Johann
Georg Mozart Joseph Lange Cäcilia Weber Josepha Weber Aloysia Weber Sophie Weber Georg Nissen Family trees Influences Beethoven Catholic Church Freemasonry
Haydn Salieri Related Mozart in popular culture Book ( hide ) Symphonies by Wolfgang Amadeus Mozart Unnumbered K. 16a ( Odense ) K. 19a K. 45a ( Old Lambach )
K. 76 K. 45b K. 74g K. 81 K. 95 K. 97 K.
</item 4>

<item 5>
Symphony No. 40 (Mozart)
Symphony No. 40 ( Mozart ) - wikipedia Symphony No. 40 ( Mozart ) Jump to : navigation , search Symphony No. 40 in G minor , KV. 550 was written by Wolfgang
Amadeus Mozart in 1788 . It is sometimes referred to as the `` Great G minor symphony '' , to distinguish it from the `` Little G minor symphony '' , No. 25 .
The two are the only extant minor key symphonies Mozart wrote . Contents ( hide ) 1 Composition 2 Premiere 3 The music 4 Reception 5 Influence 6 First recording
7 Media 8 Notes 9 References 10 External links Composition ( edit ) The date of completion of this symphony is known exactly , since Mozart in his mature years
kept a full catalog of his completed works ; he entered the 40th Symphony into it on 25 July 1788 . Work on the symphony occupied an exceptionally productive
period of just a few weeks during which time he also completed the 39th and 41st symphonies ( 26 June and 10 August , respectively ) .
</item 5>

<item 6>
Symphony No. 40 (Mozart)
Zaslaw , Neal , and William Cowdery ( 1990 ) . The Compleat Mozart : A Guide to the Musical Works of Wolfgang Amadeus Mozart . New York : W.W. Norton & Company
. Extracts cited are viewable online at Google Books : ( 6 ) . External links ( edit ) Wikimedia Commons has media related to Symphony No. 40 ( Mozart ) .
Versions of the score : Sinfonie in g minor K. 550 ( 1st version ) : Score and critical report ( in German ) in the Neue Mozart - Ausgabe Sinfonie in g minor K.
550 ( 2nd version ) : Score in the Neue Mozart - Ausgabe Symphony No. 40 : Scores at the International Music Score Library Project ( IMSLP ) Kalmus miniature
editions , from the William and Gayle Cook Music Library at the Indiana University Jacobs School of Music A Public Domain version by the Columbia University
Orchestra at TEOC Classical Music Radio : Symphony No.
</item 6>

<item 7>
Symphony No. 40 (Mozart)
The work is in four movements , in the usual arrangement for a classical - style symphony ( fast movement , slow movement , minuet , fast movement ) : Molto
allegro , Andante , Menuetto . Allegretto -- Trio , Finale . Allegro assai , The first movement begins darkly , not with its first theme but with accompaniment
, played by the lower strings with divided violas . The technique of beginning a work with an accompaniment figure was later used by Mozart in his last piano
concerto ( KV. 595 ) and later became a favourite of the Romantics ( examples include the openings of Mendelssohn 's Violin Concerto and Sergei Rachmaninoff 's
Third Piano Concerto ) . The first theme is as follows . The second movement is a lyrical work in time . It is in the subdominant key of the relative major of G
minor ( B ♭ major ) : E ♭ major .
</item 7>

<item 8>
Symphony No. 40 (Mozart)
Nikolaus Harnoncourt conjectured that Mozart composed the three symphonies as a unified work , pointing , among other things , to the fact that the Symphony No.
40 , as the middle work , has no introduction ( unlike No. 39 ) and does not have a finale of the scale of No. 41 's . The 40th symphony exists in two versions
, differing primarily in that one includes parts for a pair of clarinets ( with suitable adjustments made in the other wind parts ) . Most likely , the clarinet
parts were added in a revised version . The autograph scores of both versions were acquired in the 1860s by Johannes Brahms , who later donated the manuscripts
to the Gesellschaft der Musikfreunde in Vienna , where they reside today . Premiere ( edit ) As Neal Zaslaw has pointed out , writers on Mozart have often
suggested -- or even asserted -- that Mozart never heard his 40th Symphony performed . Some commentators go further , suggesting that Mozart wrote the symphony
( and its companions , Nos.
</item 8>

<item 9>
Symphony No. 40 (Mozart)
504 , Prague ) No. 39 ( K. 543 ) No. 40 ( K. 550 ) No. 41 ( K. 551 , Jupiter ) Lost K. 19b K. 66c K. 66d K. 66e K. Anh. C 11.07 K. Anh. C 11.08 No. 2 now
attributed to Leopold Mozart . No. 3 now attributed to Carl Friedrich Abel . No. 37 now attributed to Michael Haydn . List of symphonies by Wolfgang Amadeus
Mozart List of spurious / doubtful Mozart symphonies LCCN : n81018421 GND : 300111150 BNF : cb13915276b ( data ) Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Symphony_No._40_(Mozart)&oldid=825476175 '' Categories : Symphonies by Wolfgang Amadeus Mozart 1788 compositions
Compositions in G minor Hidden categories : All articles with specifically marked weasel - worded phrases Articles with specifically marked weasel - worded
phrases from
</item 9>

<item 10>
Symphony No. 40 (Mozart)
Franz Schubert likewise copied down the music of Mozart 's minuet , and the minuet of his Fifth Symphony strongly evokes Mozart 's . Zaslaw has suggested that a
passage late in Joseph Haydn 's oratorio The Seasons ( 1801 ) , a meditation on death , quotes the second movement of the 40th Symphony and was included by
Haydn as a memorial to his long - dead friend . First recording ( edit ) The first known recording of the 40th Symphony is by the Victor Recording Company and
issued in 1915 under the title `` Symphony in G Minor '' . The Victor Concert Orchestra performed under the direction of conductor Walter B. Rogers . Media (
edit ) The following files contain a digital recording of a performance of the 40th Symphony by the Fulda Symphonic Orchestra . The performance took place on
March 18 , 2001 in the Orangerie in Fulda , Germany . 1st movement Molto allegro 2nd movement Andante 3rd movement Menuetto , Allegretto - Trio 4th movement
Allegro assai Problems playing these files ? See media help .
</item 10>
=== END ITEM id=3efa1d9220e5d0ee ===
