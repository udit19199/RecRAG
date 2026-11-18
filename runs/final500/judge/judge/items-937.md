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

=== ITEM id=d5ade1d8c4688ccf ===
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
490 1995 352 NFC South 21 Philadelphia Eagles 555 591 26 . 485 1933 1,172 NFC East 22 Tennessee Titans 413 449 6 . 479 1960 868 AFC South 23 Buffalo Bills 400
460 8 . 465 1960 868 AFC East 24 Detroit Lions 544 641 32 . 460 1930 1,217 NFC North 25 Cincinnati Bengals 344 408 . 458 1968 756 AFC North 26 New York Jets 392
468 8 . 456 1960 868 AFC East 27 New Orleans Saints 338 427 5 . 442 1967 770 NFC South 28 Houston Texans 106 134 0 . 442 2002 240 AFC South 29 Jacksonville
Jaguars 155 197 0 . 440 1995 352 AFC South 30 Atlanta Falcons 341 437 6 . 439 1966 784 NFC South 31 Arizona Cardinals 542 732 40 . 428 1920 1,314 NFC West 32
Tampa Bay Buccaneers 250 393 .
</item 1>

<item 2>
NFL win–loss records
Ties are registered as a half win and a half loss when calculating the win - loss record . Best win -- loss record in division Rank Team Won Lost Tied Pct .
First NFL Season Total Games Division Dallas Cowboys 493 367 6 . 573 1960 866 NFC East Green Bay Packers 730 553 37 . 567 1921 1,320 NFC North Chicago Bears 744
568 42 . 565 1920 1,354 NFC North Miami Dolphins 439 341 . 563 1966 784 AFC East 5 New England Patriots 476 383 9 . 554 1960 868 AFC East 6 New York Giants 684
572 33 . 543 1925 1,289 NFC East 7 Denver Broncos 465 393 10 . 541 1960 868 AFC West 8 Minnesota Vikings 457 387 10 . 541 1961 854 NFC North 9 Baltimore Ravens
181 154 . 540 336 AFC North 10 San Francisco 49ers 522 450 14 .
</item 2>

<item 3>
List of National Football League records (individual)
Detroit Lions , 1992 -- 2010 Most wins against a single opponent , regular season ( and post season ) , on the road , by a starting quarterback : 14 , Tom Brady
, vs. Buffalo Bills , 2001 -- 2017 Most consecutive wins against a single opponent , regular season ( and post season ) , on the road , by a starting
quarterback : 11 , Brett Favre , Green Bay Packers vs. Chicago Bears , 1994 -- Jan. 2 , 2005 Fewest wins in a regular season by a starting quarterback who won
the Super Bowl : 0 , Doug Williams , Washington Redskins 1987 Oldest starting quarterback to win a playoff game : Tom Brady ( Jacksonville in the 2017 AFC
Finals ; 40 years 171 days ) . QB losses ( edit ) Note : These records are not listed in NFL Record and Fact Book Most career losses , regular season , by a
starting quarterback : 123 , Vinny Testaverde , 1987 -- 2007 Most career losses , post-season , by a starting quarterback : 13 , Peyton Manning , 1999 -- 2016 .
</item 3>

<item 4>
NFL win–loss records
540 336 AFC North 10 San Francisco 49ers 522 450 14 . 537 1950 986 NFC West 11 Indianapolis Colts 502 441 7 . 532 1953 950 AFC South 12 Oakland Raiders 456 401
11 . 532 1960 868 AFC West 13 Pittsburgh Steelers 601 535 20 . 529 1933 1,156 AFC North 14 Kansas City Chiefs 447 409 12 . 522 1960 868 AFC West 15 Washington
Redskins 586 572 28 . 506 1932 1,186 NFC East 16 Seattle Seahawks 325 318 . 505 1976 644 NFC West 17 Cleveland Browns 462 466 10 . 498 1950 938 AFC North 18 Los
Angeles Chargers 426 431 11 . 497 1960 868 AFC West 19 Los Angeles Rams 544 554 21 . 496 1937 1,119 NFC West 20 Carolina Panthers 172 179 . 490 1995 352 NFC
South 21 Philadelphia Eagles 555 591 26 .
</item 4>

<item 5>
List of National Football League records (individual)
Peyton Manning , Indianapolis Colts 1998 ; Derek Carr , Oakland Raiders , 2014 Most losses in a season , by a starting quarterback : 15 , Archie Manning , New
Orleans Saints , 1980 ; Jeff George , Indianapolis Colts , 1991 ; Deshone Kizer , Cleveland Browns , 2017 Most consecutive losses , by a rookie starting
quarterback : 15 , Deshone Kizer , Cleveland Browns , 2017 Most losses in a season , by a rookie starting quarterback : 15 , Deshone Kizer , Cleveland Browns ,
2017 Most consecutive losses in a season , by a starting quarterback : 15 , Deshone Kizer , Cleveland Browns , 2017 Most consecutive losses to start a season ,
by a starting quarterback : 15 , Deshone Kizer , Cleveland Browns , 2017 Most consecutive losses by a starting quarterback : 21 , Dan Pastorini , Houston Oilers
, 1972 -- 1974 Most consecutive losses by a starting quarterback , regular season , to start a career : 15 , Deshone Kizer , Cleveland Browns , 2017 Most
consecutive road losses by a starting quarterback , regular season , to start a career : 13 ,
</item 5>

<item 6>
NFL win–loss records
355 2002 AFC North 30 Detroit Lions 20 7 13 . 350 2016 NFC North 31 Kansas City Chiefs 26 9 17 . 346 2016 AFC West 32 Cincinnati Bengals 19 5 14 . 263 2015 AFC
North Notes ( edit ) Jump up ^ For example , the Browns ' regular season win -- loss record does not include a 47 -- 4 -- 3 record compiled in the All - America
Football Conference from 1946 to 1949 , nor does their playoff record include the 5 -- 0 record compiled during the same period . Likewise , the 49ers ' AAFC
regular season win -- loss record of 38 -- 14 -- 2 , nor their 1 - 1 AAFC playoff record are officially counted Jump up ^ The Patriots played as the `` Boston
Patriots '' from 1960 -- 1970 , before adopting the `` New England Patriots '' name , which they have played under since the 1971 season . The team was also
briefly known as the `` Bay State Patriots '' prior to the 1971 season .
</item 6>

<item 7>
NFL win–loss records
NFL win -- loss records - wikipedia NFL win -- loss records Jump to : navigation , search It has been suggested that NFL standings since AFL -- NFL merger be
merged into this article . ( Discuss ) Proposed since October 2017 . The following is a list of win -- loss records for each of the 32 active National Football
League ( NFL ) teams . Contents ( hide ) 1 Regular season 2 Playoffs 3 Notes 4 References Regular season ( edit ) George Halas ( right ) of the Chicago Bears ,
who have recorded the most regular season wins . The following is a listing of all 32 current National Football League ( NFL ) teams ranked by their regular
season win -- loss record percentage , accurate as of the end of the 2016 NFL regular season . For teams that previously played in the All - America Football
Conference ( AAFC ) , their AAFC win - loss records are not included as the NFL does not officially count AAFC statistics despite the 1950 NFL -- AAFC merger .
However , for teams that played in the American Football League ( AFL ) , their AFL win - loss records are included and officially counted by the NFL .
</item 7>

<item 8>
NFL win–loss records
480 AFC East 18 Philadelphia Eagles 40 19 21 . 475 2013 NFC East 19 Jacksonville Jaguars 11 5 6 . 455 2007 AFC South 20 Los Angeles Rams 43 19 24 . 442 NFC West
21 ( tie ) Arizona Cardinals 16 7 9 . 438 2015 NFC West 21 ( tie ) New Orleans Saints 16 7 9 . 438 2013 NFC South 23 Houston Texans 7 . 429 2016 AFC South 24
Tennessee Titans 33 14 19 . 424 2008 AFC South 25 Atlanta Falcons 22 9 13 . 409 2016 NFC South 26 Minnesota Vikings 47 19 28 . 404 2015 NFC North 27 Tampa Bay
Buccaneers 15 6 9 . 400 2007 NFC South 28 Los Angeles Chargers 28 11 17 . 393 2013 AFC West 29 Cleveland Browns 31 11 20 . 355 2002 AFC North 30 Detroit Lions
20 7 13 .
</item 8>

<item 9>
NFL win–loss records
389 1976 644 NFC South Playoffs ( edit ) Members of the Baltimore Ravens franchise , the team with the highest playoff win -- loss percentage , during the
Lombardi Trophy presentation following their Super Bowl XLVII victory Marvin Lewis , head coach of the Cincinnati Bengals , who have registered the lowest
playoff win - loss percentage of all time The following is a listing of all 32 current National Football League ( NFL ) teams ranked by their playoff win - loss
record percentage , accurate as of the end of Super Bowl LI . Best win -- loss record in division Team active in playoffs Rank Team Games Won Lost Pct . Last
playoff appearance Division Baltimore Ravens 23 15 8 . 652 2014 AFC North New England Patriots 51 32 19 . 627 2016 AFC East Green Bay Packers 56 34 22 . 607
2016 NFC North 4 ( tie ) Pittsburgh Steelers 60 36 24 . 600 2016 AFC North 4 ( tie ) San Francisco 49ers 50 30 20 . 600 2013 NFC West 6 Oakland Raiders 44 25 19
.
</item 9>

<item 10>
List of National Football League records (individual)
Most career road losses , post-season , by a starting quarterback : 7 , Brett Favre , 1993 -- 2009 Most career home losses , post-season , by a starting
quarterback : 6 , Peyton Manning , 1999 -- 2016 . Most consecutive losses , post-season , by a starting quarterback : 4 , Y.A. Tittle , 1957 , 1961 -- 1963 ;
Warren Moon 1991 -- 1994 ; Andy Dalton 2011 -- 2014 Most consecutive road losses , post-season , by a starting quarterback : 6 , Dave Krieg , 1983 , 1984 , 1987
, 1992 -- 1994 Most consecutive road losses by a starting quarterback : 14 , Steve DeBerg , 1978 -- 1979 Most consecutive home losses by a starting quarterback
: 10 , Archie Manning , 1979 -- 1980 ; Chris Weinke , 2001 -- 2002 , 2006 Most road losses in a season , by a rookie starting quarterback : 8 , Peyton Manning ,
Indianapolis Colts 1998 ; Derek Carr , Oakland Raiders ,
</item 10>
=== END ITEM id=d5ade1d8c4688ccf ===

=== ITEM id=99a108e5a01ab2f6 ===
Gold sentences and facts:
Sentence 0: Take action immediately upon hearing thunder . Anyone close enough to the storm to hear thunder can be struck by lightning . Avoid electrical appliances , including corded telephones . Cordless and wireless telephones are safe to use during a thunderstorm . Close and stay away from windows and doors , as glass can become a serious hazard in high wind . Do not bathe or shower , as plumbing conducts electricity . If driving , safely exit the roadway , turn on hazard lights , and park . Remain in the vehicle and avoid touching metal .
  Fact 0.0: Take action immediately upon hearing thunder.
  Fact 0.1: Anyone close enough to the storm to hear thunder can be struck by lightning.
  Fact 0.2: Avoid electrical appliances, including corded telephones.
  Fact 0.3: Cordless and wireless telephones are safe to use during a thunderstorm.
  Fact 0.4: Close and stay away from windows and doors.
  Fact 0.5: Glass can become a serious hazard in high wind.
  Fact 0.6: Do not bathe or shower.
  Fact 0.7: Plumbing conducts electricity.
  Fact 0.8: If driving, safely exit the roadway.
  Fact 0.9: If driving, turn on hazard lights.
  Fact 0.10: If driving, park.
  Fact 0.11: Remain in the vehicle and avoid touching metal.
Sentence 1: Take action immediately upon hearing thunder . Anyone close enough to the storm to hear thunder can be struck by lightning .
  Fact 1.0: Take action immediately upon hearing thunder.
  Fact 1.1: Anyone close enough to the storm to hear thunder can be struck by lightning.

Retrieved items:
<item 1>
Thunderstorm
A similar event occurred in Boscastle , England , on 16 August 2004 . Hazards ( edit ) Each year , many people are killed or seriously injured by severe
thunderstorms despite the advance warning . While severe thunderstorms are most common in the spring and summer , they can occur at just about any time of the
year . Cloud - to - ground lightning ( edit ) See also : Lightning strike and Wildfire A return stroke , cloud - to - ground lightning strike during a
thunderstorm . Cloud - to - ground lightning frequently occurs within the phenomena of thunderstorms and have numerous hazards towards landscapes and
populations . One of the more significant hazards lightning can pose is the wildfires they are capable of igniting . Under a regime of low precipitation ( LP )
thunderstorms , where little precipitation is present , rainfall can not prevent fires from starting when vegetation is dry as lightning produces a concentrated
amount of extreme heat . Direct damage caused by lightning strikes occurs on occasion . In areas with a high frequency for cloud - to - ground lightning , like
Florida , lightning causes several fatalities per year , most commonly to people working outside . Acid rain is also a frequent risk produced by lightning .
</item 1>

<item 2>
Thunderstorm
The National Weather Service ( NWS ) in the United States recommends several precautions that people should take if thunderstorms are likely to occur : Know the
names of local counties , cities , and towns , as these are how warnings are described . Monitor forecasts and weather conditions and know whether thunderstorms
are likely in the area . Be alert for natural signs of an approaching storm . Cancel or reschedule outdoor events ( to avoid being caught outdoors when a storm
hits ) . Take action early so you have time to get to a safe place . Get inside a substantial building or hard - topped metal vehicle before threatening weather
arrives . If you hear thunder , get to the safe place immediately . Avoid open areas like hilltops , fields , and beaches , and do n't be or be near the tallest
objects in an area when thunderstorms are occurring . Do n't shelter under tall or isolated trees during thunderstorms . If in the woods , put as much distance
between you and any trees during thunderstorms . If in a group , spread out so that you increase the chances for survivors who could come to the aid of any
victims from a lightning strike .
</item 2>

<item 3>
Thunderstorm
Lightningsafety.noaa.gov. 2014 - 06 - 30 . Retrieved 2014 - 08 - 20 . Jump up ^ `` NWS JetStream -- Lightning Frequently Asked Questions '' . Srh.noaa.gov. 2014
- 06 - 28 . Retrieved 2014 - 08 - 20 . Jump up ^ `` You 're not safer crouching : Six things you did n't know about lightning '' . LA Times . Retrieved 2014 -
08 - 20 . Jump up ^ National Geographic Almanac of Geography , ISBN 0 - 7922 - 3877 - X , page 75 . Jump up ^ `` How many thunderstorms occur each year ? '' .
Thunderstorms . Sky Fire Productions . Archived from the original on 2007 - 07 - 11 . Retrieved 2006 - 08 - 23 . Jump up ^ National Weather Service JetStream (
2008 - 10 - 08 ) . `` Tropical Cyclone Hazards '' . National Weather Service Southern Region Headquarters .
</item 3>

<item 4>
Thunderstorm
These storms can produce large hail and powerful tornadoes . Thunderstorms are relatively uncommon along much of the West Coast of the United States , but they
occur with greater frequency in the inland areas , particularly the Sacramento and San Joaquin Valleys of California . In spring and summer , they occur nearly
daily in certain areas of the Rocky Mountains as part of the North American Monsoon regime . In the Northeast , storms take on similar characteristics and
patterns as the Midwest , but with less frequency and severity . During the summer , air - mass thunderstorms are an almost daily occurrence over central and
southern parts of Florida . Energy ( edit ) See also : Sprite ( lightning ) , Upper - atmospheric lightning , and St. Elmo 's fire How thunderstorms launch
particle beams into space If the quantity of water that is condensed in and subsequently precipitated from a cloud is known , then the total energy of a
thunderstorm can be calculated . In a typical thunderstorm , approximately 5 × 10 kg of water vapor are lifted , and the amount of energy released when this
condenses is 10 joules .
</item 4>

<item 5>
Thunderstorm
Type Severe Cloud of origin Cumulonimbus . Part of the nature series Weather Calendar seasons Winter Spring Summer Autumn Tropical seasons Dry season Wet season
Storms Cloud Cumulonimbus cloud Arcus cloud Downburst Microburst Heat burst Dust storm Simoom Haboob Monsoon Gale Sirocco Firestorm Lightning Supercell
Thunderstorm Severe thunderstorm Thundersnow Storm surge Tornado Cyclone Mesocyclone Anticyclone Tropical cyclone ( Hurricane ) Extratropical cyclone European
windstorm Atlantic Hurricane Typhoon Derecho Landspout Dust devil Fire whirl Waterspout Winter storm Ice storm Blizzard Ground blizzard Snowsquall Precipitation
Drizzle ( Freezing drizzle ) Graupel Hail Ice pellets ( Diamond dust ) Rain ( Freezing rain ) Cloudburst Snow Rain and snow mixed Snow grains Snow roller Slush
Topics Atmospheric river Atmospheric convection Air pollution Climate Cloud Fog Cold wave Heat wave Jet stream Meteorology Severe weather Weather forecasting
Weather portal A typical thunderstorm over a field A thunderstorm , also known as an electrical storm , lightning storm , or thundershower , is a storm
characterized by the presence of lightning and its acoustic effect on the Earth 's atmosphere , known as thunder .
</item 5>

<item 6>
Thunderstorm
Radio pulses produced by cosmic rays are being used to study how electric charges develop within thunderstorms . More organized meteorological projects such as
VORTEX2 use an array of sensors , such as the Doppler on Wheels , vehicles with mounted automated weather stations , weather balloons , and unmanned aircraft to
investigate thunderstorms expected to produce severe weather . Lightning is detected remotely using sensors that detect cloud - to - ground lightning strokes
with 95 percent accuracy in detection and within 250 metres ( 820 ft ) of their point of origin . Mythology and religion ( edit ) Thunderstorms strongly
influenced many early civilizations . Greeks believed that they were battles waged by Zeus , who hurled lightning bolts forged by Hephaestus . Some American
Indian tribes associated thunderstorms with the Thunderbird , who they believed was a servant of the Great Spirit . The Norse considered thunderstorms to occur
when Thor went to fight Jötnar , with the thunder and lightning being the effect of his strikes with the hammer Mjölnir . Hinduism recognizes Indra as the god
of rain and thunderstorms .
</item 6>

<item 7>
Thunderstorm
Retrieved 2009 - 09 - 09 . Jump up ^ Charles H. Paxton ; J. Colson & N. Carlisle ( 2008 ) . `` P2. 13 Florida lightning deaths and injuries 2004 -- 2007 '' .
American Meteorological Society . Retrieved 2009 - 09 - 05 . Jump up ^ G.E. Likens ; W.C. Keene ; J.M. Miller & J.N. Galloway ( 1987 ) . `` Chemistry of
precipitation from a remote , terrestrial site in Australia '' . Journal of Geophysical Research . 92 ( 13 ) : 299 -- 314 . Bibcode : 1987JGR ... 92 ... 299R .
doi : 10.1029 / JA092iA01p00299 . Jump up ^ Joel S. Levine ; Tommy R. Augustsson ; Iris C. Andersont ; James M. Hoell Jr . & Dana A. Brewer ( 1984 ) . ``
Tropospheric sources of NOx : Lightning and biology '' . Atmospheric Environment.
</item 7>

<item 8>
Thunderstorm
Safety ( edit ) While safety and preparedness often overlap , `` thunderstorm safety '' generally refers to what people should do during and after a storm . The
American Red Cross recommends that people follow these precautions if a storm is imminent or in progress : Take action immediately upon hearing thunder . Anyone
close enough to the storm to hear thunder can be struck by lightning . Avoid electrical appliances , including corded telephones . Cordless and wireless
telephones are safe to use during a thunderstorm . Close and stay away from windows and doors , as glass can become a serious hazard in high wind . Do not bathe
or shower , as plumbing conducts electricity . If driving , safely exit the roadway , turn on hazard lights , and park . Remain in the vehicle and avoid
touching metal . The NWS stopped recommending the `` lightning crouch '' in 2008 as it does n't provide a significant level of protection and will not
significantly lower the risk of being killed or injured from a nearby lightning strike . Frequent occurrences ( edit ) See also : United States rainfall
climatology A mild thunderstorm over Niagara Falls , Ontario .
</item 8>

<item 9>
Thunderstorm
`` Tropical Cyclone Hazards '' . National Weather Service Southern Region Headquarters . Retrieved 2009 - 08 - 30 . Jump up ^ David Roth . `` Unified Surface
Analysis Manual '' ( PDF ) . Hydrometeorological Prediction Center . Retrieved 2006 - 10 - 22 . Jump up ^ Office of the Federal Coordinator for Meteorology (
2001 - 06 - 07 ) . `` National Severe Local Storms Operations Plan -- Chapter 2 '' ( PDF ) . Department of Commerce . Archived from the original ( PDF ) on 2009
- 05 - 06 . Retrieved 2006 - 08 - 23 . Jump up ^ Garner , Rob ( 26 June 2015 ) . `` Fermi Catches Antimatter - Hurling Storms '' . nasa.gov . Retrieved 19 July
2016 . Jump up ^ Ouellette , Jennifer ( January 13 , 2011 ) . `` Fermi Spots Antimatter in Thunderstorms '' . Discovery News . Retrieved 16 January 2011 .
</item 9>

<item 10>
Thunderstorm
doi : 10.1111 / cea. 12709 . PMID 26765082 ^ Jump up to : American Red Cross . `` Thunderstorm Safety Checklist '' ( PDF ) . American Red Cross . Retrieved 24
August 2011 . ^ Jump up to : National Weather Service Weather Forecast Office . `` Thunderstorm '' . Severe Weather Preparedness Information . Albuquerque , NM
: NOAA . Retrieved 24 August 2011 . Jump up ^ Federal Emergency Management Agency . `` Thunderstorms and Lightning '' . Ready . US Department of Homeland
Security . Archived from the original on 23 June 2011 . Retrieved 24 August 2011 . ^ Jump up to : Federal Emergency Management Agency . `` What to Do Before a
Thunderstorm '' . US Department of Homeland Security . Archived from the original on 20 August 2011 . Retrieved 24 August 2011 . ^ Jump up to : `` NWS Lightning
Safety Myths '' . Lightningsafety.noaa.gov. 2014 - 06 - 30 .
</item 10>
=== END ITEM id=99a108e5a01ab2f6 ===

=== ITEM id=bdfacd3af4e81e06 ===
Gold sentences and facts:
Sentence 0: NCAA Men 's National Collegiate Volleyball Championship Sport College indoor volleyball Founded 1970 No. of teams 6 ( as of 2014 ) Country United States Most recent champion ( s ) Ohio State 2 ( 3 ) TV partner ( s ) ESPNU Official website NCAA.com
  Fact 0.0: The NCAA Men's National Collegiate Volleyball Championship is a college indoor volleyball event.
  Fact 0.1: The NCAA Men's National Collegiate Volleyball Championship was founded in 1970.
  Fact 0.2: The NCAA Men's National Collegiate Volleyball Championship had 6 teams as of 2014.
  Fact 0.3: The NCAA Men's National Collegiate Volleyball Championship is held in the United States.
  Fact 0.4: Ohio State is listed as the most recent champion, with a count of 2 (3).
  Fact 0.5: The TV partner of the NCAA Men's National Collegiate Volleyball Championship is ESPNU.
  Fact 0.6: The official website of the NCAA Men's National Collegiate Volleyball Championship is NCAA.com.
Sentence 1: NCAA Men 's National Collegiate Volleyball Championship Year Site ( Host ) Host Arena Final Third Place Final / Other participants Winner Score Runner - up Third Place Score Fourth Place 1970 Details Los Angeles ( UCLA ) Pauley Pavilion UCLA ( 24 -- 1 ) 3 -- 0 Long Beach State UC Santa Barbara 2 -- 0 Ball State 1971 Details UCLA ( 29 -- 1 ) 3 -- 2 UC Santa Barbara Ball State 2 -- 0 Springfield 1972 Details Muncie , Indiana ( Ball State ) Irving Gymnasium UCLA ( 27 -- 7 ) 3 -- 2 San Diego State Ball State 2 -- 0 UC Santa Barbara 1973 Details San Diego ( San Diego State ) Peterson Gym San Diego State ( 21 -- 5 ) 3 -- 1 Long Beach State Ball State 2 -- 0 Army Details Santa Barbara , California ( UCSB ) Robertson Gymnasium UCLA ( 30 -- 5 ) 3 -- 2 UC Santa Barbara Ball State 2 -- 1 Springfield Details Los Angeles ( UCLA ) Pauley Pavilion UCLA ( 27 -- 8 ) 3 -- 1 UC Santa Barbara Ohio State 2 -- 0 Yale 1976 Details Muncie , Indiana ( Ball State ) Irving Gymnasium UCLA ( 15 -- 2 ) 3 -- 0 Pepperdine Ohio State 2 -- 0 Springfield 1977 Details Los Angeles ( UCLA ) Pauley Pavilion USC ( 18 -- 1 ) 3 -- 1 Ohio State Pepperdine 2 -- 0 Rutgers -- Newark 1978 Details Columbus , Ohio ( Ohio State ) St. John Arena Pepperdine ( 21 -- 4 ) 3 -- 2 UCLA Ohio State 2 -- 0 Rutgers -- Newark 1979 Details Los Angeles ( UCLA ) Pauley Pavilion UCLA ( 30 -- 0 ) 3 -- 1 USC Rutgers -- Newark 3 -- 2 Ball State 1980 Details Muncie , Indiana ( Ball State ) Irving Gymnasium USC ( 22 -- 6 ) 3 -- 1 UCLA Ohio State 3 -- 0 Rutgers -- Newark 1981 Details Santa Barbara , California ( UCSB ) UCSB Events Center UCLA ( 32 -- 3 ) 3 -- 2 USC Penn State 3 -- 1 Ohio State 1982 Details University Park , Pennsylvania ( Penn State ) Rec Hall UCLA ( 29 -- 0 ) 3 -- 0 Penn State USC 2 -- 1 Ohio State Details Columbus , Ohio ( Ohio State ) St. John Arena UCLA ( 27 -- 4 ) 3 -- 0 Pepperdine Ohio State 3 -- 1 Penn State 1984 Details Los Angeles ( UCLA ) Pauley Pavilion UCLA ( 38 -- 0 ) 3 -- 1 Pepperdine George Mason 3 -- 0 Ball State 1985 Details Pepperdine ( 25 -- 2 ) 3 -- 2 USC George Mason 3 -- 0 Ball State 1986 Details University Park , Pennsylvania ( Penn State ) Rec Hall Pepperdine ( 22 -- 7 ) 3 -- 2 USC Penn State 3 -- 0 Ohio State Details Los Angeles ( UCLA ) Pauley Pavilion UCLA ( 28 -- 3 ) 3 -- 0 USC Penn State 3 -- 0 Ohio State 1988 Details Fort Wayne , Indiana ( IPFW ) ACWMC USC ( 34 -- 4 ) 3 -- 2 UC Santa Barbara Ball State 3 -- 1 George Mason 1989 Details Los Angeles ( UCLA ) Pauley Pavilion UCLA ( 29 -- 5 ) 3 -- 1 Stanford Penn State 3 -- 0 Ball State 1990 Details Fairfax , Virginia ( George Mason ) Patriot Center USC ( 26 -- 7 ) 3 -- 1 Long Beach State Ball State 3 -- 1 Rutgers -- Newark 1991 Details Honolulu , HI ( Hawaii ) Neal S. Blaisdell Center Long Beach State ( 31 -- 4 ) 3 -- 1 USC IPFW 3 -- 1 Penn State 1992 Details Muncie , Indiana ( Ball State ) John E. Worthen Arena Pepperdine ( 24 -- 4 ) 3 -- 0 Stanford Penn State 3 -- 0 IPFW 1993 Details Los Angeles ( UCLA ) Pauley Pavilion UCLA ( 24 -- 3 ) 3 -- 0 Cal State Northridge Penn State 3 -- 2 Ohio State 1994 Details Fort Wayne , Indiana ( IPFW ) ACWMC Penn State ( 26 -- 3 ) 3 -- 2 UCLA Ball State 3 -- 0 IPFW 1995 Details Springfield , Massachusetts ( Springfield ) Springfield Civic Center UCLA ( 31 -- 1 ) 3 -- 0 Penn State Ball State 3 -- 1 Hawaii Details Los Angeles ( UCLA ) Pauley Pavilion UCLA ( 26 -- 5 ) 3 -- 2 Hawaii Lewis and Penn State 1997 Details Columbus , Ohio ( Ohio State ) St. John Arena Stanford ( 27 -- 4 ) 3 -- 2 UCLA Ball State and Penn State 1998 Details Honolulu , HI ( Hawaii ) Stan Sheriff Center UCLA ( 28 -- 4 ) 3 -- 0 Pepperdine Lewis and Princeton 1999 Details Los Angeles ( UCLA ) Pauley Pavilion BYU ( 30 -- 1 ) 3 -- 0 Long Beach State IPFW and Penn State 2000 Details Fort Wayne , Indiana ( IPFW ) ACWMC UCLA ( 29 -- 5 ) 3 -- 0 Ohio State Penn State and Pepperdine 2001 Details Long Beach , California ( Long Beach State ) The Pyramid BYU ( 23 -- 4 ) 3 -- 0 UCLA Ohio State and Penn State 2002 Details University Park , Pennsylvania ( Penn State ) Rec Hall Hawaii ( 24 -- 8 ) † 3 -- 1 Pepperdine Ball State and Penn State 2003 Details Long Beach , California ( Long Beach State ) The Pyramid Lewis ( 29 -- 6 ) † 3 -- 2 BYU Penn State and Pepperdine Details Honolulu , HI ( Hawaii ) Stan Sheriff Center BYU ( 29 -- 4 ) 3 -- 2 Long Beach State Lewis and Penn State 2005 Details Los Angeles ( UCLA ) Pauley Pavilion Pepperdine ( 25 -- 2 ) 3 -- 2 UCLA Ohio State and Penn State 2006 Details University Park , Pennsylvania ( Penn State ) Rec Hall UCLA ( 26 -- 12 ) 3 -- 0 Penn State UC Irvine and IPFW 2007 Details Columbus , Ohio ( Ohio State ) St. John Arena UC Irvine ( 29 -- 5 ) 3 -- 1 IPFW Penn State and Pepperdine 2008 Details Irvine , California ( UC Irvine ) Bren Events Center Penn State ( 30 -- 1 ) 3 -- 1 Pepperdine Long Beach State and Ohio State 2009 Details Provo , Utah ( BYU ) Smith Fieldhouse UC Irvine ( 27 -- 5 ) 3 -- 2 USC ( 21 -- 11 ) Ohio State and Penn State Details Stanford , California ( Stanford ) Maples Pavilion Stanford ( 24 -- 6 ) 3 -- 0 Penn State ( 24 -- 8 ) Cal State Northridge and Ohio State 2011 Details University Park , Pennsylvania ( Penn State ) Rec Hall Ohio State ( 26 -- 6 ) 3 -- 2 UC Santa Barbara ( 18 -- 15 ) Penn State and USC 2012 Details Los Angeles ( USC ) Galen Center UC Irvine ( 26 -- 5 ) 3 -- 0 USC ( 24 -- 6 ) Lewis and Penn State 2013 Details Los Angeles ( UCLA ) Pauley Pavilion UC Irvine ( 25 -- 7 ) 3 -- 0 BYU ( 26 -- 5 ) Loyola Chicago and Penn State 2014 Details Chicago ( Loyola Chicago ) Gentile Arena Loyola Chicago ( 29 - 1 ) 3 - 1 Stanford ( 24 - 9 ) 3rd -- BYU and Penn State 5th -- Lewis and Erskine 2015 Details Stanford , California ( Stanford ) Maples Pavilion Loyola Chicago ( 28 - 2 ) 3 - 2 Lewis ( 27 - 4 ) 3rd -- UC Irvine and Penn State 5th -- Hawaii and Pfeiffer 2016 Details University Park , Pennsylvania ( Penn State ) Rec Hall Ohio State ( 31 - 2 ) 3 - 0 BYU ( 27 - 4 ) 3rd -- UCLA and Long Beach State 5th -- Erskine and George Mason 2017 Details Columbus , Ohio ( Ohio State ) St. John Arena Ohio State ( 32 -- 2 ) 3 - 0 BYU ( 26 - 5 ) 3rd -- Hawaii and Long Beach State 5th -- Barton and Penn State 2018 Details Los Angeles ( UCLA ) Pauley Pavilion 2019 Details Long Beach , California ( Long Beach State ) The Pyramid 2020 Details Fairfax , Virginia ( George Mason ) EagleBank Arena 2021 Details Columbus , Ohio ( Ohio State ) Covelli Arena 2022 Details Los Angeles ( UCLA ) Pauley Pavilion
  Fact 1.0: In 1970, the championship was held in Los Angeles (host UCLA) at Pauley Pavilion.
  Fact 1.1: In 1970, UCLA (24-1) won the final 3-0 over runner-up Long Beach State.
  Fact 1.2: In 1970, third place went to UC Santa Barbara, with a third-place score of 2-0.
  Fact 1.3: In 1970, fourth place went to Ball State.
  Fact 1.4: In 1971, UCLA (29-1) won the final 3-2 over runner-up UC Santa Barbara.
  Fact 1.5: In 1971, third place went to Ball State, with a third-place score of 2-0.
  Fact 1.6: In 1971, fourth place went to Springfield.
  Fact 1.7: In 1972, the championship was held in Muncie, Indiana (host Ball State) at Irving Gymnasium.
  Fact 1.8: In 1972, UCLA (27-7) won the final 3-2 over runner-up San Diego State.
  Fact 1.9: In 1972, third place went to Ball State, with a third-place score of 2-0.
  Fact 1.10: In 1972, fourth place went to UC Santa Barbara.
  Fact 1.11: In 1973, the championship was held in San Diego (host San Diego State) at Peterson Gym.
  Fact 1.12: In 1973, San Diego State (21-5) won the final 3-1 over runner-up Long Beach State.
  Fact 1.13: In 1973, third place went to Ball State, with a third-place score of 2-0.
  Fact 1.14: In 1973, fourth place went to Army.
  Fact 1.15: In an unlabeled row after 1973, the championship was held in Santa Barbara, California (host UCSB) at Robertson Gymnasium.
  Fact 1.16: In that unlabeled row, UCLA (30-5) won the final 3-2 over runner-up UC Santa Barbara.
  Fact 1.17: In that unlabeled row, third place went to Ball State, with a third-place score of 2-1.
  Fact 1.18: In that unlabeled row, fourth place went to Springfield.
  Fact 1.19: In an unlabeled row after the Santa Barbara row, the championship was held in Los Angeles (host UCLA) at Pauley Pavilion.
  Fact 1.20: In that unlabeled row, UCLA (27-8) won the final 3-1 over runner-up UC Santa Barbara.
  Fact 1.21: In that unlabeled row, third place went to Ohio State, with a third-place score of 2-0.
  Fact 1.22: In that unlabeled row, fourth place went to Yale.
  Fact 1.23: In 1976, the championship was held in Muncie, Indiana (host Ball State) at Irving Gymnasium.
  Fact 1.24: In 1976, UCLA (15-2) won the final 3-0 over runner-up Pepperdine.
  Fact 1.25: In 1976, third place went to Ohio State, with a third-place score of 2-0.
  Fact 1.26: In 1976, fourth place went to Springfield.
  Fact 1.27: In 1977, the championship was held in Los Angeles (host UCLA) at Pauley Pavilion.
  Fact 1.28: In 1977, USC (18-1) won the final 3-1 over runner-up Ohio State.
  Fact 1.29: In 1977, third place went to Pepperdine, with a third-place score of 2-0.
  Fact 1.30: In 1977, fourth place went to Rutgers-Newark.
  Fact 1.31: In 1978, the championship was held in Columbus, Ohio (host Ohio State) at St. John Arena.
  Fact 1.32: In 1978, Pepperdine (21-4) won the final 3-2 over runner-up UCLA.
  Fact 1.33: In 1978, third place went to Ohio State, with a third-place score of 2-0.
  Fact 1.34: In 1978, fourth place went to Rutgers-Newark.
  Fact 1.35: In 1979, the championship was held in Los Angeles (host UCLA) at Pauley Pavilion.
  Fact 1.36: In 1979, UCLA (30-0) won the final 3-1 over runner-up USC.
  Fact 1.37: In 1979, third place went to Rutgers-Newark, with a third-place score of 3-2.
  Fact 1.38: In 1979, fourth place went to Ball State.
  Fact 1.39: In 1980, the championship was held in Muncie, Indiana (host Ball State) at Irving Gymnasium.
  Fact 1.40: In 1980, USC (22-6) won the final 3-1 over runner-up UCLA.
  Fact 1.41: In 1980, third place went to Ohio State, with a third-place score of 3-0.
  Fact 1.42: In 1980, fourth place went to Rutgers-Newark.
  Fact 1.43: In 1981, the championship was held in Santa Barbara, California (host UCSB) at the UCSB Events Center.
  Fact 1.44: In 1981, UCLA (32-3) won the final 3-2 over runner-up USC.
  Fact 1.45: In 1981, third place went to Penn State, with a third-place score of 3-1.
  Fact 1.46: In 1981, fourth place went to Ohio State.
  Fact 1.47: In 1982, the championship was held in University Park, Pennsylvania (host Penn State) at Rec Hall.
  Fact 1.48: In 1982, UCLA (29-0) won the final 3-0 over runner-up Penn State.
  Fact 1.49: In 1982, third place went to USC, with a third-place score of 2-1.
  Fact 1.50: In 1982, fourth place went to Ohio State.
  Fact 1.51: In an unlabeled row, the championship was held in Columbus, Ohio (host Ohio State) at St. John Arena.
  Fact 1.52: In that unlabeled row, UCLA (27-4) won the final 3-0 over runner-up Pepperdine.
  Fact 1.53: In that unlabeled row, third place went to Ohio State, with a third-place score of 3-1.
  Fact 1.54: In that unlabeled row, fourth place went to Penn State.
  Fact 1.55: In 1984, the championship was held in Los Angeles (host UCLA) at Pauley Pavilion.
  Fact 1.56: In 1984, UCLA (38-0) won the final 3-1 over runner-up Pepperdine.
  Fact 1.57: In 1984, third place went to George Mason, with a third-place score of 3-0.
  Fact 1.58: In 1984, fourth place went to Ball State.
  Fact 1.59: In 1985, Pepperdine (25-2) won the final 3-2 over runner-up USC.
  Fact 1.60: In 1985, third place went to George Mason, with a third-place score of 3-0.
  Fact 1.61: In 1985, fourth place went to Ball State.
  Fact 1.62: In 1986, the championship was held in University Park, Pennsylvania (host Penn State) at Rec Hall.
  Fact 1.63: In 1986, Pepperdine (22-7) won the final 3-2 over runner-up USC.
  Fact 1.64: In 1986, third place went to Penn State, with a third-place score of 3-0.
  Fact 1.65: In 1986, fourth place went to Ohio State.
  Fact 1.66: In an unlabeled row, the championship was held in Los Angeles (host UCLA) at Pauley Pavilion.
  Fact 1.67: In that unlabeled row, UCLA (28-3) won the final 3-0 over runner-up USC.
  Fact 1.68: In that unlabeled row, third place went to Penn State, with a third-place score of 3-0.
  Fact 1.69: In that unlabeled row, fourth place went to Ohio State.
  Fact 1.70: In 1988, the championship was held in Fort Wayne, Indiana (host IPFW) at ACWMC.
  Fact 1.71: In 1988, USC (34-4) won the final 3-2 over runner-up UC Santa Barbara.
  Fact 1.72: In 1988, third place went to Ball State, with a third-place score of 3-1.
  Fact 1.73: In 1988, fourth place went to George Mason.
  Fact 1.74: In 1989, the championship was held in Los Angeles (host UCLA) at Pauley Pavilion.
  Fact 1.75: In 1989, UCLA (29-5) won the final 3-1 over runner-up Stanford.
  Fact 1.76: In 1989, third place went to Penn State, with a third-place score of 3-0.
  Fact 1.77: In 1989, fourth place went to Ball State.
  Fact 1.78: In 1990, the championship was held in Fairfax, Virginia (host George Mason) at Patriot Center.
  Fact 1.79: In 1990, USC (26-7) won the final 3-1 over runner-up Long Beach State.
  Fact 1.80: In 1990, third place went to Ball State, with a third-place score of 3-1.
  Fact 1.81: In 1990, fourth place went to Rutgers-Newark.
  Fact 1.82: In 1991, the championship was held in Honolulu, HI (host Hawaii) at Neal S. Blaisdell Center.
  Fact 1.83: In 1991, Long Beach State (31-4) won the final 3-1 over runner-up USC.
  Fact 1.84: In 1991, third place went to IPFW, with a third-place score of 3-1.
  Fact 1.85: In 1991, fourth place went to Penn State.
  Fact 1.86: In 1992, the championship was held in Muncie, Indiana (host Ball State) at John E. Worthen Arena.
  Fact 1.87: In 1992, Pepperdine (24-4) won the final 3-0 over runner-up Stanford.
  Fact 1.88: In 1992, third place went to Penn State, with a third-place score of 3-0.
  Fact 1.89: In 1992, fourth place went to IPFW.
  Fact 1.90: In 1993, the championship was held in Los Angeles (host UCLA) at Pauley Pavilion.
  Fact 1.91: In 1993, UCLA (24-3) won the final 3-0 over runner-up Cal State Northridge.
  Fact 1.92: In 1993, third place went to Penn State, with a third-place score of 3-2.
  Fact 1.93: In 1993, fourth place went to Ohio State.
  Fact 1.94: In 1994, the championship was held in Fort Wayne, Indiana (host IPFW) at ACWMC.
  Fact 1.95: In 1994, Penn State (26-3) won the final 3-2 over runner-up UCLA.
  Fact 1.96: In 1994, third place went to Ball State, with a third-place score of 3-0.
  Fact 1.97: In 1994, fourth place went to IPFW.
  Fact 1.98: In 1995, the championship was held in Springfield, Massachusetts (host Springfield) at Springfield Civic Center.
  Fact 1.99: In 1995, UCLA (31-1) won the final 3-0 over runner-up Penn State.
  Fact 1.100: In 1995, third place went to Ball State, with a third-place score of 3-1.
  Fact 1.101: In 1995, fourth place went to Hawaii.
  Fact 1.102: In an unlabeled row between 1995 and 1997, the championship was held in Los Angeles (host UCLA) at Pauley Pavilion.
  Fact 1.103: In that unlabeled row, UCLA (26-5) won the final 3-2 over runner-up Hawaii.
  Fact 1.104: In that unlabeled row, Lewis and Penn State are listed as the remaining finishers, with no score.
  Fact 1.105: In 1997, the championship was held in Columbus, Ohio (host Ohio State) at St. John Arena.
  Fact 1.106: In 1997, Stanford (27-4) won the final 3-2 over runner-up UCLA.
  Fact 1.107: In 1997, Ball State and Penn State are listed as the remaining finishers, with no score.
  Fact 1.108: In 1998, the championship was held in Honolulu, HI (host Hawaii) at Stan Sheriff Center.
  Fact 1.109: In 1998, UCLA (28-4) won the final 3-0 over runner-up Pepperdine.
  Fact 1.110: In 1998, Lewis and Princeton are listed as the remaining finishers, with no score.
  Fact 1.111: In 1999, the championship was held in Los Angeles (host UCLA) at Pauley Pavilion.
  Fact 1.112: In 1999, BYU (30-1) won the final 3-0 over runner-up Long Beach State.
  Fact 1.113: In 1999, IPFW and Penn State are listed as the remaining finishers, with no score.
  Fact 1.114: In 2000, the championship was held in Fort Wayne, Indiana (host IPFW) at ACWMC.
  Fact 1.115: In 2000, UCLA (29-5) won the final 3-0 over runner-up Ohio State.
  Fact 1.116: In 2000, Penn State and Pepperdine are listed as the remaining finishers, with no score.
  Fact 1.117: In 2001, the championship was held in Long Beach, California (host Long Beach State) at The Pyramid.
  Fact 1.118: In 2001, BYU (23-4) won the final 3-0 over runner-up UCLA.
  Fact 1.119: In 2001, Ohio State and Penn State are listed as the remaining finishers, with no score.
  Fact 1.120: In 2002, the championship was held in University Park, Pennsylvania (host Penn State) at Rec Hall.
  Fact 1.121: In 2002, Hawaii (24-8), marked with a dagger, won the final 3-1 over runner-up Pepperdine.
  Fact 1.122: In 2002, Ball State and Penn State are listed as the remaining finishers, with no score.
  Fact 1.123: In 2003, the championship was held in Long Beach, California (host Long Beach State) at The Pyramid.
  Fact 1.124: In 2003, Lewis (29-6), marked with a dagger, won the final 3-2 over runner-up BYU.
  Fact 1.125: In 2003, Penn State and Pepperdine are listed as the remaining finishers, with no score.
  Fact 1.126: In an unlabeled row between 2003 and 2005, the championship was held in Honolulu, HI (host Hawaii) at Stan Sheriff Center.
  Fact 1.127: In that unlabeled row, BYU (29-4) won the final 3-2 over runner-up Long Beach State.
  Fact 1.128: In that unlabeled row, Lewis and Penn State are listed as the remaining finishers, with no score.
  Fact 1.129: In 2005, the championship was held in Los Angeles (host UCLA) at Pauley Pavilion.
  Fact 1.130: In 2005, Pepperdine (25-2) won the final 3-2 over runner-up UCLA.
  Fact 1.131: In 2005, Ohio State and Penn State are listed as the remaining finishers, with no score.
  Fact 1.132: In 2006, the championship was held in University Park, Pennsylvania (host Penn State) at Rec Hall.
  Fact 1.133: In 2006, UCLA (26-12) won the final 3-0 over runner-up Penn State.
  Fact 1.134: In 2006, UC Irvine and IPFW are listed as the remaining finishers, with no score.
  Fact 1.135: In 2007, the championship was held in Columbus, Ohio (host Ohio State) at St. John Arena.
  Fact 1.136: In 2007, UC Irvine (29-5) won the final 3-1 over runner-up IPFW.
  Fact 1.137: In 2007, Penn State and Pepperdine are listed as the remaining finishers, with no score.
  Fact 1.138: In 2008, the championship was held in Irvine, California (host UC Irvine) at Bren Events Center.
  Fact 1.139: In 2008, Penn State (30-1) won the final 3-1 over runner-up Pepperdine.
  Fact 1.140: In 2008, Long Beach State and Ohio State are listed as the remaining finishers, with no score.
  Fact 1.141: In 2009, the championship was held in Provo, Utah (host BYU) at Smith Fieldhouse.
  Fact 1.142: In 2009, UC Irvine (27-5) won the final 3-2 over runner-up USC (21-11).
  Fact 1.143: In 2009, Ohio State and Penn State are listed as the remaining finishers, with no score.
  Fact 1.144: In an unlabeled row between 2009 and 2011, the championship was held in Stanford, California (host Stanford) at Maples Pavilion.
  Fact 1.145: In that unlabeled row, Stanford (24-6) won the final 3-0 over runner-up Penn State (24-8).
  Fact 1.146: In that unlabeled row, Cal State Northridge and Ohio State are listed as the remaining finishers, with no score.
  Fact 1.147: In 2011, the championship was held in University Park, Pennsylvania (host Penn State) at Rec Hall.
  Fact 1.148: In 2011, Ohio State (26-6) won the final 3-2 over runner-up UC Santa Barbara (18-15).
  Fact 1.149: In 2011, Penn State and USC are listed as the remaining finishers, with no score.
  Fact 1.150: In 2012, the championship was held in Los Angeles (host USC) at Galen Center.
  Fact 1.151: In 2012, UC Irvine (26-5) won the final 3-0 over runner-up USC (24-6).
  Fact 1.152: In 2012, Lewis and Penn State are listed as the remaining finishers, with no score.
  Fact 1.153: In 2013, the championship was held in Los Angeles (host UCLA) at Pauley Pavilion.
  Fact 1.154: In 2013, UC Irvine (25-7) won the final 3-0 over runner-up BYU (26-5).
  Fact 1.155: In 2013, Loyola Chicago and Penn State are listed as the remaining finishers, with no score.
  Fact 1.156: In 2014, the championship was held in Chicago (host Loyola Chicago) at Gentile Arena.
  Fact 1.157: In 2014, Loyola Chicago (29-1) won the final 3-1 over runner-up Stanford (24-9).
  Fact 1.158: In 2014, BYU and Penn State are listed as third place.
  Fact 1.159: In 2014, Lewis and Erskine are listed as fifth place.
  Fact 1.160: In 2015, the championship was held in Stanford, California (host Stanford) at Maples Pavilion.
  Fact 1.161: In 2015, Loyola Chicago (28-2) won the final 3-2 over runner-up Lewis (27-4).
  Fact 1.162: In 2015, UC Irvine and Penn State are listed as third place.
  Fact 1.163: In 2015, Hawaii and Pfeiffer are listed as fifth place.
  Fact 1.164: In 2016, the championship was held in University Park, Pennsylvania (host Penn State) at Rec Hall.
  Fact 1.165: In 2016, Ohio State (31-2) won the final 3-0 over runner-up BYU (27-4).
  Fact 1.166: In 2016, UCLA and Long Beach State are listed as third place.
  Fact 1.167: In 2016, Erskine and George Mason are listed as fifth place.
  Fact 1.168: In 2017, the championship was held in Columbus, Ohio (host Ohio State) at St. John Arena.
  Fact 1.169: In 2017, Ohio State (32-2) won the final 3-0 over runner-up BYU (26-5).
  Fact 1.170: In 2017, Hawaii and Long Beach State are listed as third place.
  Fact 1.171: In 2017, Barton and Penn State are listed as fifth place.
  Fact 1.172: In 2018, the championship was held in Los Angeles (host UCLA) at Pauley Pavilion.
  Fact 1.173: In 2019, the championship was held in Long Beach, California (host Long Beach State) at The Pyramid.
  Fact 1.174: In 2020, the championship was held in Fairfax, Virginia (host George Mason) at EagleBank Arena.
  Fact 1.175: In 2021, the championship was held in Columbus, Ohio (host Ohio State) at Covelli Arena.
  Fact 1.176: In 2022, the championship was held in Los Angeles (host UCLA) at Pauley Pavilion.

Retrieved items:
<item 1>
NCAA Men's National Collegiate Volleyball Championship
Stanford , California Championship Saturday , May 9 , 2015 Maples Pavilion , Stanford , California Lewis ( 3 ) 25 22 25 25 Hawaii ( 1 ) 22 20 25 25 5 Penn State
( 1 ) 20 25 16 20 5 Penn State ( 3 ) 25 25 17 27 Lewis ( 2 ) 25 23 15 27 21 Loyola ( Chicago ) ( 3 ) 21 25 25 25 23 UC Irvine ( 0 ) 22 19 17 Loyola ( Chicago )
( 3 ) 25 33 25 Loyola ( Chicago ) ( 3 ) 25 25 25 6 Pfeiffer ( 0 ) 20 31 15 2016 ( edit ) Main article : 2016 NCAA Men 's National Collegiate Volleyball
Tournament See also ( edit ) NCAA Division III Men 's Volleyball Championship NCAA Women 's Volleyball Championship List of NCAA men 's volleyball schools
Pre-NCAA Intercollegiate Volleyball Champions Notes ( edit ) Jump up ^ Athletic Business ,
</item 1>

<item 2>
NCAA Men's National Collegiate Volleyball Championship
Pennsylvania ( Penn State ) Rec Hall Ohio State ( 26 -- 6 ) 3 -- 2 UC Santa Barbara ( 18 -- 15 ) Penn State and USC 2012 Details Los Angeles ( USC ) Galen
Center UC Irvine ( 26 -- 5 ) 3 -- 0 USC ( 24 -- 6 ) Lewis and Penn State 2013 Details Los Angeles ( UCLA ) Pauley Pavilion UC Irvine ( 25 -- 7 ) 3 -- 0 BYU ( 26
-- 5 ) Loyola Chicago and Penn State 2014 Details Chicago ( Loyola Chicago ) Gentile Arena Loyola Chicago ( 29 - 1 ) 3 - 1 Stanford ( 24 - 9 ) 3rd -- BYU and
Penn State 5th -- Lewis and Erskine 2015 Details Stanford , California ( Stanford ) Maples Pavilion Loyola Chicago ( 28 - 2 ) 3 - 2 Lewis ( 27 - 4 ) 3rd -- UC
Irvine and Penn State 5th -- Hawaii and Pfeiffer 2016 Details University Park ,
</item 2>

<item 3>
NCAA Men's National Collegiate Volleyball Championship
California - based universities have dominated the men 's volleyball national championship ; Loyola Chicago , Penn State , Ohio State and BYU are the only
non-California universities to have won the National Collegiate championship ; Hawaii and Lewis also won the championship tournament , but had their victories
vacated due to NCAA rules violations . Only seven non-California universities have participated in the National Collegiate championship match ( Loyola , BYU ,
Penn State , Ohio State , IPFW , Hawaii , and Lewis ) , although other universities such as Princeton and Ball State have participated in the final four . Only
three finals have involved two non-California schools : the 2003 final , when Lewis defeated BYU but had its win vacated ; the 2015 final , in which Loyola
defeated Lewis ; and the 2016 final , when Ohio State defeated BYU . UCLA , Southern California , Penn State , Stanford , and Long Beach State are the only
schools in Division I to have won an NCAA national championship in both men and women 's volleyball .
</item 3>

<item 4>
NCAA Men's National Collegiate Volleyball Championship
Pennsylvania ( Penn State ) Rec Hall Ohio State ( 31 - 2 ) 3 - 0 BYU ( 27 - 4 ) 3rd -- UCLA and Long Beach State 5th -- Erskine and George Mason 2017 Details
Columbus , Ohio ( Ohio State ) St. John Arena Ohio State ( 32 -- 2 ) 3 - 0 BYU ( 26 - 5 ) 3rd -- Hawaii and Long Beach State 5th -- Barton and Penn State 2018
Details Los Angeles ( UCLA ) Pauley Pavilion 2019 Details Long Beach , California ( Long Beach State ) The Pyramid 2020 Details Fairfax , Virginia ( George
Mason ) EagleBank Arena 2021 Details Columbus , Ohio ( Ohio State ) Covelli Arena 2022 Details Los Angeles ( UCLA ) Pauley Pavilion † Vacated due to NCAA
violations Team titles ( edit ) UCLA Pepperdine USC UC Irvine BYU Penn State Stanford Loyola Ohio State ⇙ Hawaii Lewis Long Beach State San Diego State Schools
with national championships -- 19 championships , -- 5 championships , -- 4 championships , -- 3 championships , -- 2 championships ,
</item 4>

<item 5>
NCAA Men's National Collegiate Volleyball Championship
Gender Equity -- Boys ' and Mens ' Volleyball Participation Continues to Lag , April 2009 Jump up ^
http://fs.ncaa.org/Docs/stats/m_volleyball_champs_records/2012/d1/2012d1mvbchamp.pdf Jump up ^ UC Irvine defeats BYU in three sets to claim consecutive national
title , NCAA.com , May 5 , 2013 Jump up ^ Loyola repeats as men 's NCAA volleyball national champions , Chicago Tribune , May 10 , 2015 Jump up ^ NCAA Tourney
to have 2 play - in matches starting in 2014 , Off the Block , October 23 , 2013 NCAA Men 's National Collegiate Volleyball Championship Tournaments 1970 1971
1972 1973 1976 1977 1978 1979 1980 1981 1982 1984 1985 1986 1988 1989 1990 1991 1992 1993 1994 1995 1997 1998 1999 2000 2001 2002 2003 2005 2006
</item 5>

<item 6>
NCAA Men's National Collegiate Volleyball Championship
May 2 , 2013 Pauley Pavilion , Los Angeles Championship Saturday , May 4 , 2013 Pauley Pavilion , Los Angeles Brigham Young ( 3 ) 25 25 25 Penn State ( 0 ) 21
16 22 Brigham Young ( 0 ) 23 22 24 UC Irvine ( 3 ) 25 25 26 UC Irvine ( 3 ) 26 25 29 Loyola University Chicago ( 0 ) 24 18 27 2014 ( edit ) The semifinals and
finals 2014 tournament were held in the Gentile Arena in Chicago on the campus of Loyola University Chicago . Two quarterfinal `` play - in '' matches were held
at the Gentile Arena two days prior to the national semifinals , as the 2014 tournament expanded to six teams for the first time ever . A second at - large was
added to the field , and the champions of the newly - eligible Conference Carolinas men 's volleyball division got an automatic qualification .
</item 6>

<item 7>
NCAA Men's National Collegiate Volleyball Championship
Champions ( edit ) NCAA Men 's National Collegiate Volleyball Championship Year Site ( Host ) Host Arena Final Third Place Final / Other participants Winner
Score Runner - up Third Place Score Fourth Place 1970 Details Los Angeles ( UCLA ) Pauley Pavilion UCLA ( 24 -- 1 ) 3 -- 0 Long Beach State UC Santa Barbara 2
-- 0 Ball State 1971 Details UCLA ( 29 -- 1 ) 3 -- 2 UC Santa Barbara Ball State 2 -- 0 Springfield 1972 Details Muncie , Indiana ( Ball State ) Irving
Gymnasium UCLA ( 27 -- 7 ) 3 -- 2 San Diego State Ball State 2 -- 0 UC Santa Barbara 1973 Details San Diego ( San Diego State ) Peterson Gym San Diego State (
21 -- 5 ) 3 -- 1 Long Beach State Ball State 2 -- 0 Army Details Santa Barbara ,
</item 7>

<item 8>
NCAA Men's National Collegiate Volleyball Championship
-- 5 championships , -- 4 championships , -- 3 championships , -- 2 championships , -- 1 championship School # Year won UCLA 19 1970 , 1971 , 1972 , 1974 , 1975
, 1976 , 1979 , 1981 , 1982 , 1983 , 1984 , 1987 , 1989 , 1993 , 1995 , 1996 , 1998 , 2000 , 2006 Pepperdine 5 1978 , 1985 , 1986 , 1992 , 2005 Southern
California 1977 , 1980 , 1988 , 1990 UC Irvine 2007 , 2009 , 2012 , 2013 Ohio State 2011 , 2016 , 2017 BYU 1999 , 2001 , 2004 Penn State 1994 , 2008 Stanford
1997 , 2010 Loyola Chicago 2014 , 2015 Vacated 2002 , 2003 Long Beach State 1991 San Diego State 1973 Past tournaments ( edit ) Historically ,
</item 8>

<item 9>
NCAA Men's National Collegiate Volleyball Championship
California ( Long Beach State ) The Pyramid BYU ( 23 -- 4 ) 3 -- 0 UCLA Ohio State and Penn State 2002 Details University Park , Pennsylvania ( Penn State ) Rec
Hall Hawaii ( 24 -- 8 ) † 3 -- 1 Pepperdine Ball State and Penn State 2003 Details Long Beach , California ( Long Beach State ) The Pyramid Lewis ( 29 -- 6 ) †
3 -- 2 BYU Penn State and Pepperdine Details Honolulu , HI ( Hawaii ) Stan Sheriff Center BYU ( 29 -- 4 ) 3 -- 2 Long Beach State Lewis and Penn State 2005
Details Los Angeles ( UCLA ) Pauley Pavilion Pepperdine ( 25 -- 2 ) 3 -- 2 UCLA Ohio State and Penn State 2006 Details University Park , Pennsylvania ( Penn
State ) Rec Hall UCLA ( 26 -- 12 ) 3 -- 0 Penn State UC Irvine and IPFW 2007 Details Columbus , Ohio ( Ohio State ) St. John Arena UC Irvine ( 29 -- 5 ) 3 -- 1
IPFW Penn State and Pepperdine 2008 Details Irvine ,
</item 9>

<item 10>
NCAA Men's National Collegiate Volleyball Championship
( But for Hawaii 's vacated 2002 title , it would have matched this feat with 3 NCAA and 1 AIAW women 's national championships . ) In addition , Stanford (
1996 -- 97 ) and Penn State ( 2007 -- 08 ) are the only universities whose men and women 's volleyball programs won the national championship in the same
academic year . The 2016 championship tournament took place from May 3 -- 7 at Rec Hall on the campus of Penn State . 2011 ( edit ) May 5 , 2011 -- UC Santa
Barbara def . Southern California , 29 - 27 , 24 - 26 , 25 - 15 , 25 - 18 ; Ohio State def . Penn St. , 25 - 18 , 24 - 26 , 25 - 22 , 25 - 23 May 7 , 2011 --
Ohio State def .
</item 10>
=== END ITEM id=bdfacd3af4e81e06 ===

=== ITEM id=c850aa8b16ee2d39 ===
Gold sentences and facts:
Sentence 0: The eighth and final season of the fantasy drama television series Game of Thrones was announced by HBO in July 2016 . Unlike the first six seasons that each had ten episodes and the seventh that had seven episodes , the eighth season will have only six episodes . Like the previous season , it will largely consist of original content not found currently in George R.R. Martin 's A Song of Ice and Fire series , and will instead adapt material Martin has revealed to showrunners about the upcoming novels in the series , The Winds of Winter and A Dream of Spring .
  Fact 0.0: The eighth and final season of the fantasy drama television series Game of Thrones was announced by HBO in July 2016.
  Fact 0.1: The first six seasons of Game of Thrones each had ten episodes.
  Fact 0.2: The seventh season of Game of Thrones had seven episodes.
  Fact 0.3: The eighth season of Game of Thrones will have only six episodes.
  Fact 0.4: Like the previous season, the eighth season will largely consist of original content not found currently in George R.R. Martin's A Song of Ice and Fire series.
  Fact 0.5: The eighth season will instead adapt material Martin has revealed to showrunners about the upcoming novels in the series.
  Fact 0.6: The upcoming novels in the series are The Winds of Winter and A Dream of Spring.

Retrieved items:
<item 1>
Game of Thrones (season 8)
Queen ( 2013 ) The Rogue Prince ( 2014 ) The Sons of the Dragon ( 2017 ) TV series Episodes Season 1 5 6 7 8 Awards Characters Music `` Game of Thrones Theme ''
`` The Rains of Castamere '' `` The Bear and the Maiden Fair '' `` Light of the Seven '' Catch the Throne Live Concert Experience Soundtracks Season 1 5 6 7
Title sequence After the Thrones Thronecast Video games A Game of Thrones : Genesis Game of Thrones ( 2012 ) Game of Thrones Ascent Game of Thrones ( 2014 )
Other media The World of Ice & Fire ( 2014 ) Board game Card game ( Second edition ) Comic book series Characters Petyr Baelish Joffrey Baratheon Renly
Baratheon Robert Baratheon Stannis Baratheon Tommen Baratheon Ramsay Bolton Roose Bolton Bronn Gregor Clegane Sandor Clegane Khal Drogo Gendry Tormund
Giantsbane Gilly Theon Greyjoy Cersei Lannister Jaime Lannister Tyrion Lannister Tywin Lannister Oberyn Martell Melisandre Missandei Jorah
</item 1>

<item 2>
Game of Thrones (season 8)
Benioff and Weiss spoke about the end of the show , saying , `` From the beginning we 've wanted to tell a 70 - hour movie . It will turn out to be a 73 - hour
movie , but it 's stayed relatively the same of having the beginning , middle and now we 're coming to the end . It would have been really tough if we lost any
core cast members along the way , I 'm very happy we 've kept everyone and we get to finish it the way we want to . '' The season is set to air in 2019 . Music
( edit ) Ramin Djawadi is set to return as the composer of the show for the eighth season . References ( edit ) Jump up ^ Hibberd , James ( July 30 , 2016 ) .
`` Game of Thrones : HBO confirms season 8 will be last '' . Entertainment Weekly . Archived from the original on September 1 , 2016 . Retrieved March 13 , 2017
. Jump up ^ Birnbaum , Debra ( July 30 , 2016 ) .
</item 2>

<item 3>
Game of Thrones (season 8)
Game of Thrones ( season 8 ) - wikipedia Game of Thrones ( season 8 ) Jump to : navigation , search Game of Thrones ( season 8 ) Promotional poster Starring See
List of Game of Thrones cast Country of origin United States Release Original network HBO Season chronology ← Previous Season 7 List of Game of Thrones episodes
The eighth and final season of the fantasy drama television series Game of Thrones was announced by HBO in July 2016 . Unlike the first six seasons that each
had ten episodes and the seventh that had seven episodes , the eighth season will have only six episodes . Like the previous season , it will largely consist of
original content not found currently in George R.R. Martin 's A Song of Ice and Fire series , and will instead adapt material Martin has revealed to showrunners
about the upcoming novels in the series , The Winds of Winter and A Dream of Spring . The season will be adapted for television by David Benioff and D.B. Weiss
. Filming officially began on October 23 , 2017 . The season is scheduled to premiere in 2019 .
</item 3>

<item 4>
Game of Thrones (season 8)
Jump up ^ Stolworthy , Jacob ( January 17 , 2018 ) . `` Game of Thrones season 8 : Melisandre return confirmed as fans theorise on character storyline '' . The
Independent . Retrieved January 25 , 2018 . Jump up ^ Nolan , Emma ( January 21 , 2018 ) . `` Game of Thrones season 8 cast : Who will be in the final series ?
'' . Daily Express . Retrieved March 25 , 2018 . Jump up ^ Deen , Sarah ( March 24 , 2018 ) . `` Game Of Thrones star Joe Dempsie is scared he 'll get drunk and
reveal final season secrets '' . Metro . Retrieved March 26 , 2018 . Jump up ^ McCreesh , Louise ( November 6 , 2017 ) . `` The Night 's Watch WILL be in Game
of Thrones season 8 , confirms Lord Commander actor '' . Digital Spy . Archived from the original on December 6 , 2017 . Retrieved December 18 , 2017 .
</item 4>

<item 5>
Game of Thrones (season 8)
The final episode of the show will be directed by Benioff and Weiss , who have previously directed one episode each . At the show 's South by Southwest panel on
March 12 , 2017 , Benioff and Weiss announced the writers for the show to be Dave Hill ( episode 1 ) and Bryan Cogman ( episode 2 ) . The showrunners will then
divide up the screenplay for the remaining four episodes amongst themselves . Writing ( edit ) Writing for the eighth season started with a 140 - page outline .
Benioff said that the divvying up process and who should write what section became more difficult , being that `` this would be the last time that we would be
doing this . '' Filming ( edit ) In an interview with Entertainment Weekly , HBO programming president Casey Bloys stated that instead of the series finale
being a feature film , the final season would be `` six one - hour movies '' on television . He continued , `` The show has proven that TV is every bit as
impressive and in many cases more so , than film . What they 're doing is monumental . ''
</item 5>

<item 6>
Game of Thrones (season 8)
Jump up ^ Chitwood , Adam ( December 15 , 2017 ) . `` ' Justice League ' DP Fabian Wagner on Zack Snyder 's Cut , Superman 's Black Suit & ' Game of Thrones '
'' . Collider . Archived from the original on December 16 , 2017 . Retrieved December 18 , 2017 . Jump up ^ Carson , Biz ( March 12 , 2017 ) . `` The final
season of ' Game of Thrones ' will only be six episodes '' . Business Insider . Archived from the original on April 12 , 2017 . Retrieved April 11 , 2017 . ^
Jump up to : `` Game of Thrones Will Return in 2019 '' . HBO . Archived from the original on January 4 , 2018 . Retrieved January 4 , 2018 . ^ Jump up to :
Goldberg , Lesley ( June 21 , 2016 ) . `` ' Game of Thrones ' Stars Score Hefty Pay Raises for Season 8 '' . The Hollywood Reporter .
</item 6>

<item 7>
Game of Thrones (season 8)
Jump up ^ Birnbaum , Debra ( July 30 , 2016 ) . `` HBO Confirms ' Game of Thrones ' Will End With Eighth Season '' . Variety . Archived from the original on
September 2 , 2016 . Retrieved March 13 , 2017 . Jump up ^ Hibberd , James ( May 24 , 2016 ) . `` George R.R. Martin revealed 3 huge shocks to Game of Thrones
producers '' . Entertainment Weekly . Archived from the original on August 21 , 2016 . Retrieved March 13 , 2017 . ^ Jump up to : Hooton , Christopher (
September 27 , 2017 ) . `` Game of Thrones season 8 : Directors revealed for all episodes '' . The Independent . Archived from the original on October 20 , 2017
. Retrieved October 20 , 2017 . Jump up ^ Chitwood , Adam ( December 15 , 2017 ) .
</item 7>

<item 8>
Game of Thrones (season 8)
What they 're doing is monumental . '' Filming officially began on October 23 , 2017 . Episodes ( edit ) Co-creators David Benioff and D.B. Weiss have said that
the seventh and eighth season would likely consist of fewer episodes , stating that after season six , they were `` down to our final 13 episodes after this
season . We 're heading into the final lap . '' Benioff and Weiss stated that they were unable to produce 10 episodes in the show 's usual 12 to 14 month time
frame , as Weiss said , `` It 's crossing out of a television schedule into more of a mid-range movie schedule . '' HBO confirmed in July 2016 , that the
seventh season would consist of seven episodes , and would premiere later than usual in mid-2017 because of the later filming schedule . Benioff and Weiss later
confirmed that the eighth season will consist of six episodes , and is expected to premiere later than usual for the same reason .
</item 8>

<item 9>
Game of Thrones (season 8)
`` ' Game of Thrones ' Creators Mull Shorter Final Seasons ( EXCLUSIVE ) '' . Variety . Archived from the original on August 21 , 2016 . Retrieved June 27 ,
2016 . Jump up ^ Goldberg , Lesley ( April 14 , 2016 ) . `` ' Game of Thrones ' Considering Shorter Final Seasons '' . The Hollywood Reporter . Archived from
the original on June 18 , 2016 . Retrieved June 27 , 2016 . Jump up ^ Hibberd , James ( July 18 , 2016 ) . `` Game of Thrones : HBO announces summer return , 7
episodes '' . Entertainment Weekly . Archived from the original on September 1 , 2016 . Retrieved July 18 , 2016 . ^ Jump up to : Hibberd , James ( March 12 ,
2017 ) . `` Game of Thrones producers confirm final season only 6 episodes '' . Entertainment Weekly . Archived from the original on March 12 , 2017 . Retrieved
March 13 , 2017 .
</item 9>

<item 10>
Game of Thrones (season 8)
Weiss 2019 ( 2019 ) TBD 73 6 TBA David Benioff & D.B. Weiss David Benioff & D.B. Weiss 2019 ( 2019 ) TBD Cast ( edit ) See also : List of Game of Thrones
characters Peter Dinklage ( Tyrion Lannister ) Nikolaj Coster - Waldau ( Jaime Lannister ) Lena Headey ( Cersei Lannister ) Emilia Clarke ( Daenerys Targaryen )
Main cast ( edit ) Peter Dinklage as Tyrion Lannister Nikolaj Coster - Waldau as Jaime Lannister Lena Headey as Cersei Lannister Emilia Clarke as Daenerys
Targaryen Kit Harington as Jon Snow Liam Cunningham as Davos Seaworth Sophie Turner as Sansa Stark Maisie Williams as Arya Stark Nathalie Emmanuel as Missandei
Gwendoline Christie as Brienne of Tarth John Bradley as Samwell Tarly Isaac Hempstead Wright as Bran Stark Rory McCann as Sandor `` The Hound '' Clegane Iain
Glen as Jorah Mormont Conleth Hill as Varys
</item 10>
=== END ITEM id=c850aa8b16ee2d39 ===

=== ITEM id=07113f3cd7cfbd8d ===
Gold sentences and facts:
Sentence 0: Rigor mortis is very important in meat technology . The onset of rigor mortis and its resolution partially determine the tenderness of meat . If the post-slaughter meat is immediately chilled to 15 ° C ( 59 ° F ) , a phenomenon known as cold shortening occurs , whereby the muscle sarcomeres shrink to a third of their original length .
  Fact 0.0: Rigor mortis is very important in meat technology
  Fact 0.1: The onset of rigor mortis partially determines the tenderness of meat
  Fact 0.2: The resolution of rigor mortis partially determines the tenderness of meat
  Fact 0.3: If post-slaughter meat is immediately chilled to 15 °C (59 °F), a phenomenon known as cold shortening occurs
  Fact 0.4: Cold shortening is the phenomenon that occurs when post-slaughter meat is immediately chilled to 15 °C (59 °F)
  Fact 0.5: In cold shortening, the muscle sarcomeres shrink to a third of their original length

Retrieved items:
<item 1>
Rigor mortis
One such factor is the ambient temperature . When conditions are warm , the onset and pace of rigor mortis are sped up by providing a conducive environment for
the metabolic processes that cause decay . Low temperatures , however , slow them down . Therefore , for a person who dies outside in frozen conditions rigor
mortis may last several days more than normal , so investigators may have to abandon it as a tool for determining time of death . See also ( edit ) Cadaveric
spasm References ( edit ) ^ Jump up to : Saladin , K.S. 2010 . Anatomy & Physiology : 6th edition . McGraw - Hill . ^ Jump up to : Hall , John E. , and Arthur
C. Guyton . Guyton and Hall Textbook of Medical Physiology . Philadelphia , PA : Saunders / Elsevier , 2011 . MD Consult . Web . 26 Jan. 2015 . Jump up ^
Fremery , Donald ( 3 February 1959 ) . `` Biochemistry of Chicken Muscle as Related to Rigor Mortis and Tenderization '' . Journal of Food Sciences . 25 ( 1 ) :
73 -- 87 .
</item 1>

<item 2>
Rigor mortis
2011 . Web . 4 December 2011 . < http://www.deathreference.com/Py-Se/Rigor-Mortis-and-Other-Postmortem-Changes.html > . Saladin , Kenneth . Anatomy and
Physiology : The Unity of Form and Function , 6th ed . McGraw - Hill . New York , 2012 . Peress , Robin . `` Discovery Health `` Rigor Mortis at the Crime Scene
'' '' Discovery Health `` Health Guides '' Discovery Fit & Health , 2011 . Web . 4 December 2011 . <
http://health.howstuffworks.com/diseases-conditions/death-dying/rigor-mortis-cause2.htm > . Death Outline In medicine Cell death Necrosis Avascular necrosis
Coagulative necrosis Liquefactive necrosis Gangrenous necrosis Caseous necrosis Fat necrosis Fibrinoid necrosis Temporal lobe necrosis Programmed cell death
AICD Anoikis Apoptosis Autophagy Intrinsic apoptosis Necroptosis Paraptosis Parthanatos Phenoptosis Pseudoapoptosis Pyroptosis Autolysis
</item 2>

<item 3>
Rigor mortis
The calcium ions trigger powerful muscle contraction aided by ATP molecules . To prevent cold shortening , a process known as electrical stimulation is carried
out , especially in beef carcasses , immediately after slaughter and skinning . In this process , the carcass is stimulated with alternating current , causing
it to contract and relax , which depletes the ATP reserve from the carcass and prevents cold shortening . Application in forensic pathology ( edit ) The degree
of rigor mortis may be used in forensic pathology , to determine the approximate time of death . A dead body holds its position as rigor mortis sets in . If the
body is moved after death , but before rigor mortis begins , forensic techniques such as livor mortis can be applied . If the position in which a body is found
does not match the location where it is found ( for example , if it is flat on its back with one arm sticking straight up ) , that could mean someone moved it .
Several factors also affect the progression of rigor mortis , and investigators take these into account when estimating the time of death . One such factor is
the ambient temperature .
</item 3>

<item 4>
Rigor mortis
The sequence may be due to different lactic acid levels among different muscles , which is directly related to the difference in glycogen levels and different
types of muscle fibers . Rigor mortis then spreads to the other muscles , including the internal organs , within the next four to six hours . The onset of rigor
mortis is affected by the individual 's age , sex , physical condition , and muscular build . Rigor mortis may not be perceivable in many infant and child
corpses due to their smaller muscle mass . Applications in meat industry ( edit ) Rigor mortis is very important in meat technology . The onset of rigor mortis
and its resolution partially determine the tenderness of meat . If the post-slaughter meat is immediately chilled to 15 ° C ( 59 ° F ) , a phenomenon known as
cold shortening occurs , whereby the muscle sarcomeres shrink to a third of their original length . Cold shortening is caused by the release of stored calcium
ions from the sarcoplasmic reticulum of muscle fibers , in response to the cold stimulus . The calcium ions trigger powerful muscle contraction aided by ATP
molecules .
</item 4>

<item 5>
Rigor mortis
Rigor mortis - wikipedia Rigor mortis Jump to : navigation , search This article is about the sign of death . For other uses , see Rigor mortis ( disambiguation
) . Stages of death Pallor mortis Algor mortis Rigor mortis Livor mortis Putrefaction Decomposition Skeletonization Fossilization Rigor mortis ( Latin : rigor
`` stiffness '' , mortis `` of death '' ) or postmortem rigidity , the third stage of death , is one of the recognizable signs of death , caused by chemical
changes in the muscles post mortem , which cause the limbs of the corpse to stiffen . In humans , rigor mortis can occur as soon as 4 hours post mortem .
Contents ( hide ) 1 Physiology 2 Physical changes 3 Applications in meat industry 4 Application in forensic pathology 5 See also 6 References 7 Bibliography
Physiology ( edit ) After death , respiration in an organism ceases , depleting the source of oxygen used in the making of adenosine triphosphate ( ATP ) .
</item 5>

<item 6>
Rigor mortis
Journal of Food Sciences . 25 ( 1 ) : 73 -- 87 . doi : 10.1111 / j. 1365 - 2621.1960. tb17938. x . Retrieved 2 December 2015 . Jump up ^ About.com Jump up ^ ``
Classroom Resources - Argonne National Laboratory '' . Jump up ^ `` Rigor Mortis and Other Postmortem Changes - Burial , Body , Life , Cause , Time , Person ,
Human , Putrefaction . '' Encyclopedia of Death and Dying. 2011 . Web . 4 December 2011 . <
http://www.deathreference.com/Py-Se/Rigor-Mortis-and-Other-Postmortem-Changes.html > . Jump up ^ `` New Zealand Journal of Agricultural Research '' . Jump up ^
Peress , Robin . `` Discovery Health `` Rigor Mortis at the Crime Scene '' '' Discovery Health `` Health Guides '' Discovery Fit & Health , 2011 . Web . 4
December 2011 .
</item 6>

<item 7>
Rigor mortis
In rigor mortis myosin heads continue binding with the active sites of actin proteins via adenosine diphosphate ( ADP ) , and the muscle is unable to relax
until further enzyme activity degrades the complex . Normal relaxation would occur by replacing ADP with ATP , which would destabilize the myosin - actin bond
and break the cross-bridge . However , as ATP is absent , there must be a breakdown of muscle tissue by enzymes ( endogenous or bacterial ) during decomposition
. As part of the process of decomposition , the myosin heads are degraded by the enzymes , allowing the muscle contraction to release and the body to relax .
Decomposition of the myofilaments occurs 48 to 60 hours after the peak of rigor mortis , which occurs approximately 13 hours after death . Physical Changes (
edit ) At the time of death , a condition called `` primary flaccidity '' occurs . Following this , the muscles stiffen in rigor mortis . All muscles in the
body are affected . Starting between two and six hours following death , rigor mortis begins with the eyelids , neck , and jaw .
</item 7>

<item 8>
Rigor mortis
Web . 4 December 2011 . < http://health.howstuffworks.com/diseases-conditions/death-dying/rigor-mortis-cause2.htm > Jump up ^ Estimating The Time of Death ,
ExploreForensics < http://www.exploreforensics.co.uk/estimating-the-time-of-death.html Bibliography ( edit ) Bear , Mark F ; Connors , Barry W. ; Paradiso ,
Michael A. , Neuroscience , Exploring the Brain , Philadelphia : Lippincott Williams & Wilkins ; Third Edition ( 1 February 2006 ) . ISBN 0 - 7817 - 6003 - 8
Robert G. Mayer , `` Embalming : history , theory , and practice '' , McGraw - Hill Professional , 2005 , ISBN 0 - 07 - 143950 - 1 `` Rigor Mortis and Other
Postmortem Changes - Burial , Body , Life , Cause , Time , Person , Human , Putrefaction . '' Encyclopedia of Death and Dying. 2011 . Web . 4 December 2011 .
</item 8>

<item 9>
Rigor mortis
chemistry Post-mortem photography Taphonomy Biostratinomy Thanatology Other Apparent death Dark tourism Darwin Awards Dead pool Death and culture Death
anniversary Death anxiety Death deity Personification of death Dying - and - rising god Psychopomp Death camp Death drive Death education Death from laughter
Death hoax Death knell Death march Death messenger Death notification Death panel Death poem Death pose Death - positive movement Death squad Death threat Death
trajectory Dignified death Extinction Fan death Festival of the Dead Fascination with death Homicide Information - theoretic death Last rites Martyr Megadeath
Museum of Death Necronym Necrophilia Necrophobia The Order of the Good Death Predation Sacrifice human animal Spiritual death Suicide Assisted suicide
Thanatosensitivity Undead Voodoo death Portal WikiProject Retrieved from `` https://en.wikipedia.org/w/index.php?title=Rigor_mortis&oldid=837973659 ''
Categories : Signs of death Latin medical phrases Medical aspects of death Forensic pathology Hidden categories : All articles with dead external links Articles
with dead external links from April 2018 Articles with permanently dead external links Use dmy dates from April 2012 Talk Contents About Wikipedia Afrikaans
Беларуская
</item 9>

<item 10>
Rigor mortis
By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit
organization . About Wikipedia
</item 10>
=== END ITEM id=07113f3cd7cfbd8d ===

=== ITEM id=c4a68221cd0dc0dd ===
Gold sentences and facts:
Sentence 0: The Dead Sea ( Hebrew : יָם הַמֶּלַח ‬ Yam ha - Melah lit . Sea of Salt ; Arabic : البحر الميت ‎ Al - Bahr al - Mayyit ) is a salt lake bordered by Jordan to the east and Israel and Palestine to the west . Its surface and shores are 430.5 metres ( 1,412 ft ) below sea level , Earth 's lowest elevation on land . The Dead Sea is 304 m ( 997 ft ) deep , the deepest hypersaline lake in the world . With a salinity of 342 g / kg , or 34.2 % , ( in 2011 ) , it is 9.6 times as salty as the ocean and one of the world 's saltiest bodies of water . This salinity makes for a harsh environment in which plants and animals can not flourish , hence its name . The Dead Sea is 50 kilometres ( 31 mi ) long and 15 kilometres ( 9 mi ) wide at its widest point . It lies in the Jordan Rift Valley , and its main tributary is the Jordan River .
  Fact 0.0: The Dead Sea is also called יָם הַמֶּלַח (Yam ha-Melah) in Hebrew, which means Sea of Salt.
  Fact 0.1: The Dead Sea is also called البحر الميت (Al-Bahr al-Mayyit) in Arabic.
  Fact 0.2: The Dead Sea is a salt lake.
  Fact 0.3: The Dead Sea is bordered by Jordan to the east.
  Fact 0.4: The Dead Sea is bordered by Israel and Palestine to the west.
  Fact 0.5: The surface and shores of the Dead Sea are 430.5 metres (1,412 ft) below sea level.
  Fact 0.6: The surface and shores of the Dead Sea are Earth's lowest elevation on land.
  Fact 0.7: The Dead Sea is 304 m (997 ft) deep.
  Fact 0.8: The Dead Sea is the deepest hypersaline lake in the world.
  Fact 0.9: In 2011, the Dead Sea had a salinity of 342 g/kg, or 34.2%.
  Fact 0.10: The Dead Sea is 9.6 times as salty as the ocean.
  Fact 0.11: The Dead Sea is one of the world's saltiest bodies of water.
  Fact 0.12: The Dead Sea's salinity makes for a harsh environment in which plants and animals cannot flourish.
  Fact 0.13: The Dead Sea's name comes from its harsh environment in which plants and animals cannot flourish.
  Fact 0.14: The Dead Sea is 50 kilometres (31 mi) long.
  Fact 0.15: The Dead Sea is 15 kilometres (9 mi) wide at its widest point.
  Fact 0.16: The Dead Sea lies in the Jordan Rift Valley.
  Fact 0.17: The main tributary of the Dead Sea is the Jordan River.

Retrieved items:
<item 1>
Dead Sea
The Greeks called it Lake Asphaltites ( Attic Greek ἡ Θάλαττα ἀσφαλτῖτης , hē Thálatta asphaltĩtēs , `` the Asphaltite sea '' ) . Geography Satellite photograph
showing the location of the Dead Sea east of the Mediterranean Sea The Dead Sea is an endorheic lake located in the Jordan Rift Valley , a geographic feature
formed by the Dead Sea Transform ( DST ) . This left lateral - moving transform fault lies along the tectonic plate boundary between the African Plate and the
Arabian Plate . It runs between the East Anatolian Fault zone in Turkey and the northern end of the Red Sea Rift offshore of the southern tip of Sinai . It is
here that the Upper Jordan River / Sea of Galilee / Lower Jordan River water system comes to an end . The Jordan River is the only major water source flowing
into the Dead Sea , although there are small perennial springs under and around the Dead Sea , forming pools and quicksand pits along the edges . There are no
outlet streams .
</item 1>

<item 2>
Dead Sea
In January 2015 it was reported that the level of water is now dropping by 1 m ( 3 ft ) a year . On 27 November 2016 , it was announced that the Jordanian
government is shortlisting five consortiums to implement the project . Jordan 's ministry of Water and Irrigation said that the $100 million first phase of the
project will begin construction in the first quarter of 2018 , and will be completed by 2021 . Views in 1972 , 1989 , and 2011 compared See also Red Sea -- Dead
Sea Water Conveyance Aral Sea Dead Sea Works Great Salt Lake List of drying lakes List of places on land with elevations below sea level Mediterranean -- Dead
Sea Canal Benjamin Elazari Volcani World Discoveries III : Dead Sea References ^ Jump up to : `` Virtual Israel Experience : The Dead Sea '' . Jewish Virtual
Library . Retrieved 21 January 2013 . ^ Jump up to : `` The Dead Sea Is Dying Fast : Is It Too Late to Save It , or Was It Always a Lost Cause ? '' . Haaretz. 7
October 2016 .
</item 2>

<item 3>
Dead Sea
depth 298 m ( 978 ft ) ( elevation of deepest point , 728 m BSL ( below sea level ) , minus current surface elevation ) Water volume 114 km ( 27 cu mi ) Shore
length 135 km ( 84 mi ) Surface elevation − 430.5 m ( − 1,412 ft ) ( 2016 ) References Shore length is not a well - defined measure . Play media Short video
about the Dead Sea from the Israeli News Company The Dead Sea ( Hebrew : יָם הַמֶּלַח ‬ Yam ha - Melah lit . Sea of Salt ; Arabic : البحر الميت ‎ Al - Bahr al -
Mayyit ) is a salt lake bordered by Jordan to the east and Israel and Palestine to the west . Its surface and shores are 430.5 metres ( 1,412 ft ) below sea
level , Earth 's lowest elevation on land . The Dead Sea is 304 m ( 997 ft ) deep , the deepest hypersaline lake in the world .
</item 3>

<item 4>
The Dead Sea is a landlocked, hypersaline lake in the Jordan Rift Valley, bordered by Jordan, Israel and the West Bank; its shore is Earth’s lowest land
elevation. Fed chiefly by the Jordan River, it has shrunk sharply as river water is diverted, falling about a metre yearly and triggering sinkholes and erosion.
Red Sea–Dead Sea schemes aim to supply water and stabilize it. The region is rich in minerals, history and tourism; its salts and climate are studied for health
</item 4>

<item 5>
Dead Sea
With a salinity of 342 g / kg , or 34.2 % , ( in 2011 ) , it is 9.6 times as salty as the ocean and one of the world 's saltiest bodies of water . This salinity
makes for a harsh environment in which plants and animals can not flourish , hence its name . The Dead Sea is 50 kilometres ( 31 mi ) long and 15 kilometres ( 9
mi ) wide at its widest point . It lies in the Jordan Rift Valley , and its main tributary is the Jordan River . The Dead Sea has attracted visitors from around
the Mediterranean basin for thousands of years . It was one of the world 's first health resorts ( for Herod the Great ) , and it has been the supplier of a
wide variety of products , from asphalt for Egyptian mummification to potash for fertilisers . People also use the salt and the minerals from the Dead Sea to
create cosmetics and herbal sachets . The Dead Seawater has a density of 1.24 kg / litre , which makes swimming similar to floating . The Dead Sea is receding
at an alarming rate .
</item 5>

<item 6>
The Dead Sea is a landlocked hypersaline lake between Jordan and Israel/Palestine, in the Jordan Rift Valley. Its surface lies about 430 m below sea level—the
lowest land elevation—and it is roughly 50 km long, 15 km wide, and up to 304 m deep. With salinity around 34%, it is among the world’s saltiest lakes, making
people unusually buoyant and limiting aquatic life. Fed mainly by the Jordan River, it has a long
</item 6>

<item 7>
Dead Sea
Dead Sea - wikipedia Dead Sea Jump to : navigation , search For the novel , see Dead Sea ( novel ) . Dead Sea ים המלח ‬ ( in Hebrew ) البحر الميت ( in Arabic )
A view of the sea from the Israeli shore Location Israel Jordan Palestine Coordinates 31 ° 30 ′ N 35 ° 30 ′ E ﻿ / ﻿ 31.500 ° N 35.500 ° E ﻿ / 31.500 ; 35.500
Coordinates : 31 ° 30 ′ N 35 ° 30 ′ E ﻿ / ﻿ 31.500 ° N 35.500 ° E ﻿ / 31.500 ; 35.500 Lake type Endorheic Hypersaline Primary inflows Jordan River Primary
outflows None Catchment area 41,650 km ( 16,080 sq mi ) Basin countries Israel Jordan Palestine Max. length 50 km ( 31 mi ) ( northern basin only ) Max . width
15 km ( 9.3 mi ) Surface area 605 km ( 234 sq mi ) ( 2016 ) Average depth 199 m ( 653 ft ) Max .
</item 7>

<item 8>
Dead Sea
In Roman and Byzantine times , sugarcane , henna , and sycamore fig all made the lower Jordan valley wealthy . One of the most valuable products produced by
Jericho was the sap of the balsam tree , which could be made into perfume . By the 19th century , Jericho 's fertility had disappeared . Human settlement There
are several small communities near the Dead Sea . These include Ein Gedi , Neve Zohar and the Israeli settlements in the Megilot Regional Council : Kalya ,
Mitzpe Shalem and Avnat . There is a nature preserve at Ein Gedi , and several Dead Sea hotels are located on the southwest end at Ein Bokek near Neve Zohar .
Highway 90 runs north - south on the Israeli side for a total distance of 565 km ( 351 mi ) from Metula on the Lebanese border in the north to its southern
terminus at the Egyptian border near the Red Sea port of Eilat . Potash City is a small community on the Jordanian side of the Dead Sea , and others including
Suweima .
</item 8>

<item 9>
The Dead Sea is a hypersaline, endorheic lake in the Jordan Rift Valley, bordered by Israel, Jordan, and Palestine. Fed chiefly by the Jordan River and several
wadis and springs, it has no outlet. In 2016 it covered about 605 km²; its northern basin is up to 50 km long and 15 km wide, with an average depth of 199 m.
The lake lies along the Dead Sea Transform fault between the African and Arabian plates
</item 9>

<item 10>
The Dead Sea has rapidly receded, largely because water from the Jordan River is diverted; its level fell from 395 m below sea level in 1970 to 418 m in 2006,
at about 1 m annually. The Jordan-led Red Sea–Dead Sea Water Conveyance project aims to supply water to neighboring countries and return brine to stabilize the
sea. Its first phase was planned for 2018–2021. Dead Sea salts also attract interest for skincare products.
</item 10>
=== END ITEM id=c4a68221cd0dc0dd ===
