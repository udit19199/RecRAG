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

=== ITEM id=863855e554d8b6f2 ===
Gold sentences and facts:
Sentence 0: No . overall No. in season Title Directed by Written by Original air date Prod . code U.S. viewers ( millions ) 107 `` Watch Me '' Jann Turner Shonda Rhimes October 5 , 2017 ( 2017 - 10 - 05 ) 701 5.52 108 `` Pressing the Flesh '' Tony Goldwyn Matt Byrne October 12 , 2017 ( 2017 - 10 - 12 ) 702 5.00 109 `` Day 101 '' Scott Foley Zahir McGhee October 19 , 2017 ( 2017 - 10 - 19 ) 703 4.70 110 `` Lost Girls '' Nicole Rubio Ameni Rozsa & Austin Guzman October 26 , 2017 ( 2017 - 10 - 26 ) 704 4.88 111 5 `` Adventures in Babysitting '' Oliver Bokelberg Serveriano Canales & Tia Napolitano November 2 , 2017 ( 2017 - 11 - 02 ) 705 4.89 112 6 `` Vampires and Bloodsuckers '' Jann Turner Chris Van Dusen & Tia Napolitano November 9 , 2017 ( 2017 - 11 - 09 ) 706 5.00 113 7 `` Something Borrowed '' Sharat Raju Mark Fish November 16 , 2017 ( 2017 - 11 - 16 ) 707 4.97 114 8 `` Robin '' Daryn Okada Juan Carlos Fernandez January 18 , 2018 ( 2018 - 01 - 18 ) 709 5.17 115 9 `` Good People '' Nzingha Stewart Shonda Rhimes , Jess Brownell & Nicholas Nardini January 25 , 2018 ( 2018 - 01 - 25 ) 708 TBD 116 10 `` The People v. Olivia Pope '' TBA TBA February 1 , 2018 ( 2018 - 02 - 01 ) 710 TBD 117 11 `` Army of One '' TBA TBA February 8 , 2018 ( 2018 - 02 - 08 ) 711 TBD 118 12 TBA TBA TBA March 1 , 2018 ( 2018 - 03 - 01 ) TBA TBD
  Fact 0.0: Overall episode 107, 'Watch Me', was directed by Jann Turner and written by Shonda Rhimes.
  Fact 0.1: Overall episode 107, 'Watch Me', aired October 5, 2017, with production code 701 and 5.52 million U.S. viewers.
  Fact 0.2: Overall episode 108, 'Pressing the Flesh', was directed by Tony Goldwyn and written by Matt Byrne.
  Fact 0.3: Overall episode 108, 'Pressing the Flesh', aired October 12, 2017, with production code 702 and 5.00 million U.S. viewers.
  Fact 0.4: Overall episode 109, 'Day 101', was directed by Scott Foley and written by Zahir McGhee.
  Fact 0.5: Overall episode 109, 'Day 101', aired October 19, 2017, with production code 703 and 4.70 million U.S. viewers.
  Fact 0.6: Overall episode 110, 'Lost Girls', was directed by Nicole Rubio and written by Ameni Rozsa and Austin Guzman.
  Fact 0.7: Overall episode 110, 'Lost Girls', aired October 26, 2017, with production code 704 and 4.88 million U.S. viewers.
  Fact 0.8: Overall episode 111, season episode 5, 'Adventures in Babysitting', was directed by Oliver Bokelberg and written by Serveriano Canales and Tia Napolitano.
  Fact 0.9: Overall episode 111, 'Adventures in Babysitting', aired November 2, 2017, with production code 705 and 4.89 million U.S. viewers.
  Fact 0.10: Overall episode 112, season episode 6, 'Vampires and Bloodsuckers', was directed by Jann Turner and written by Chris Van Dusen and Tia Napolitano.
  Fact 0.11: Overall episode 112, 'Vampires and Bloodsuckers', aired November 9, 2017, with production code 706 and 5.00 million U.S. viewers.
  Fact 0.12: Overall episode 113, season episode 7, 'Something Borrowed', was directed by Sharat Raju and written by Mark Fish.
  Fact 0.13: Overall episode 113, 'Something Borrowed', aired November 16, 2017, with production code 707 and 4.97 million U.S. viewers.
  Fact 0.14: Overall episode 114, season episode 8, 'Robin', was directed by Daryn Okada and written by Juan Carlos Fernandez.
  Fact 0.15: Overall episode 114, 'Robin', aired January 18, 2018, with production code 709 and 5.17 million U.S. viewers.
  Fact 0.16: Overall episode 115, season episode 9, 'Good People', was directed by Nzingha Stewart and written by Shonda Rhimes, Jess Brownell and Nicholas Nardini.
  Fact 0.17: Overall episode 115, 'Good People', aired January 25, 2018, with production code 708 and U.S. viewer numbers to be determined.
  Fact 0.18: Overall episode 116, season episode 10, 'The People v. Olivia Pope', has a director and writers to be announced.
  Fact 0.19: Overall episode 116, 'The People v. Olivia Pope', aired February 1, 2018, with production code 710 and U.S. viewer numbers to be determined.
  Fact 0.20: Overall episode 117, season episode 11, 'Army of One', has a director and writers to be announced.
  Fact 0.21: Overall episode 117, 'Army of One', aired February 8, 2018, with production code 711 and U.S. viewer numbers to be determined.
  Fact 0.22: Overall episode 118, season episode 12, has a title, director, writers and production code to be announced.
  Fact 0.23: Overall episode 118 aired March 1, 2018, with U.S. viewer numbers to be determined.
Sentence 1: 116 10 `` The People v. Olivia Pope '' TBA TBA February 1 , 2018 ( 2018 - 02 - 01 ) 710 TBD
  Fact 1.0: Overall episode 116, season episode 10, 'The People v. Olivia Pope', has a director and writers to be announced.
  Fact 1.1: Overall episode 116, 'The People v. Olivia Pope', aired February 1, 2018, with production code 710 and U.S. viewer numbers to be determined.

Retrieved items:
<item 1>
List of Scandal episodes
Attorney David Rosen ( Joshua Malina ) ; Head of NSA , Jake Ballard ( Scott Foley ) ; chairwoman Elizabeth North ( Portia de Rossi ) ; activist Marcus Walker (
Cornelius Smith Jr . ) ; and head of B613 Eli Pope ( Joe Morton ) . As of January 18 , 2018 , 114 episodes of Scandal have aired , including one special . On
February 10 , 2017 , the show was renewed for a seventh season consisting of 18 episodes . It was later announced it will be the final season . Contents ( hide
) 1 Series overview 2 Episodes 2.1 Season 1 ( 2012 ) 2.2 Season 2 ( 2012 -- 13 ) 2.3 Season 3 ( 2013 -- 14 ) 2.4 Season 4 ( 2014 -- 15 ) 2.5 Season 5 ( 2015 --
16 ) 2.6 Season 6 ( 2017 ) 2.7 Season 7 ( 2017 -- 18 ) 3 Specials 4 Webisodes 4.
</item 1>

<item 2>
List of Scandal episodes
2017 ( 2017 - 05 - 18 ) 616 5.23 Season 7 ( 2017 -- 18 ) ( edit ) Main article : Scandal ( season 7 ) No . overall No. in season Title Directed by Written by
Original air date Prod . code U.S. viewers ( millions ) 107 `` Watch Me '' Jann Turner Shonda Rhimes October 5 , 2017 ( 2017 - 10 - 05 ) 701 5.52 108 ``
Pressing the Flesh '' Tony Goldwyn Matt Byrne October 12 , 2017 ( 2017 - 10 - 12 ) 702 5.00 109 `` Day 101 '' Scott Foley Zahir McGhee October 19 , 2017 ( 2017
- 10 - 19 ) 703 4.70 110 `` Lost Girls '' Nicole Rubio Ameni Rozsa & Austin Guzman October 26 , 2017 ( 2017 - 10 - 26 ) 704 4.
</item 2>

<item 3>
List of Scandal episodes
2012 ( 2012 - 05 - 17 ) 107 7.33 Season 2 ( 2012 -- 13 ) ( edit ) Main article : Scandal ( season 2 ) No . overall No. in season Title Directed by Written by
Original air date Prod . code U.S. viewers ( millions ) 8 `` White Hat 's Off '' Tom Verica Jenna Bans September 27 , 2012 ( 2012 - 09 - 27 ) 201 6.74 9 `` The
Other Woman '' Stephen Cragg Heather Mitchell October 4 , 2012 ( 2012 - 10 - 04 ) 202 6.56 10 `` Hunting Season '' Ron Underwood Matt Byrne October 18 , 2012 (
2012 - 10 - 18 ) 203 6.17 11 `` Beltway Unbuckled '' Mark Tinker Mark Fish October 25 , 2012 ( 2012 - 10 - 25 ) 204 6.11 12 5 `` All Roads Lead to Fitz '' Steve
Robin Raamla Mohamed November 8 ,
</item 3>

<item 4>
Graph query row:
series: Scandal
season: Season 7
episode: Pressing the Flesh
air_date: October 12, 2017
season_episode_number: None
overall_episode_number: None
kind: television episode
</item 4>

<item 5>
List of Scandal episodes
2014 ( 2014 - 04 - 17 ) 318 10.57 Season 4 ( 2014 -- 15 ) ( edit ) Main article : Scandal ( season 4 ) No . overall No. in season Title Directed by Written by
Original air date Prod . code U.S. viewers ( millions ) 48 `` Randy , Red , Superfreak and Julia '' Tom Verica Shonda Rhimes September 25 , 2014 ( 2014 - 09 -
25 ) 401 11.96 49 `` The State of the Union '' Allison Liddi - Brown Heather Mitchell October 2 , 2014 ( 2014 - 10 - 02 ) 402 10.34 50 `` Inside the Bubble ''
Randy Zisk Matt Byrne October 9 , 2014 ( 2014 - 10 - 09 ) 403 9.52 51 `` Like Father , Like Daughter '' Paul McCrane Mark Fish October 16 , 2014 ( 2014 - 10 -
16 ) 404A 9.
</item 5>

<item 6>
List of Scandal episodes
Jess Brownell & Nicholas Nardini January 25 , 2018 ( 2018 - 01 - 25 ) 708 TBD 116 10 `` The People v. Olivia Pope '' TBA TBA February 1 , 2018 ( 2018 - 02 - 01
) 710 TBD 117 11 `` Army of One '' TBA TBA February 8 , 2018 ( 2018 - 02 - 08 ) 711 TBD 118 12 TBA TBA TBA March 1 , 2018 ( 2018 - 03 - 01 ) TBA TBD Specials (
edit ) No . Title Narrator Aired between Original air date U.S.
</item 6>

<item 7>
Graph query row:
series: Scandal
season: Season 7
episode: Lost Girls
air_date: October 26, 2017
season_episode_number: None
overall_episode_number: None
kind: television episode
</item 7>

<item 8>
Graph query row:
series: Scandal (TV series)
season: Season 7
episode: Day 101
air_date: October 19, 2017
season_episode_number: None
overall_episode_number: None
kind: television episode
</item 8>

<item 9>
Graph query row:
series: Scandal
season: Season 7
episode: Day 101
air_date: October 19, 2017
season_episode_number: None
overall_episode_number: None
kind: television episode
</item 9>

<item 10>
Graph query row:
series: Scandal (TV series)
season: Season 7
episode: Lost Girls
air_date: October 26, 2017
season_episode_number: None
overall_episode_number: None
kind: television episode
</item 10>

<item 11>
List of Scandal episodes
TV by the Numbers . Retrieved January 22 , 2018 . Jump up ^ `` ( # 708 ) `` Good People '' `` . The Futon Critic . Retrieved January 24 , 2018 . ^ Jump up to :
`` Shows A-Z - scandal on abc '' . The Futon Critic . Retrieved January 24 , 2018 . Jump up ^ `` ' Scandal ' Sets Darby Stanchfield - Directed Web Series Ahead
of Season 6 '' . The Hollywood Reporter . January 18 , 2017 . External links ( edit ) Official website List of Scandal episodes on IMDb ( hide ) Scandal
Episodes Season 1 `` Sweet Baby '' Season 2 `` Nobody Likes Babies '' `` Whiskey Tango Foxtrot '' `` Boom Goes the Dynamite '' `` Top of the Hour '' `` Snake in
the Garden '' `` Molly , You in Danger , Girl '' `` Seven Fifty - Two '' Season 3 `` It 's Handled '' `` Guess Who 's Coming to Dinner '' `` Mrs.
</item 11>

<item 12>
Graph query row:
series: Scandal (TV series)
season: Season 7
episode: Watch Me
air_date: October 5, 2017
season_episode_number: None
overall_episode_number: None
kind: television episode
</item 12>

<item 13>
List of Scandal episodes
2015 ( 2015 - 09 - 24 ) May 12 , 2016 ( 2016 - 05 - 12 ) 16 10.68 6 16 January 26 , 2017 ( 2017 - 01 - 26 ) May 18 , 2017 ( 2017 - 05 - 18 ) 30 8.16 7 18
October 5 , 2017 ( 2017 - 10 - 05 ) April 19 , 2018 ( 2018 - 04 - 19 ) N / A N / A Episodes ( edit ) Season 1 ( 2012 ) ( edit ) Main article : Scandal ( season
1 ) No . overall No. in season Title Directed by Written by Original air date Prod . code U.S. viewers ( millions ) `` Sweet Baby '' Paul McGuigan Shonda Rhimes
April 5 , 2012 ( 2012 - 04 - 05 ) 101 7.33 `` Dirty Little Secrets '' Roxann Dawson Heather Mitchell April 12 ,
</item 13>

<item 14>
Graph query row:
series: Scandal (TV series)
season: Season 7
episode: Pressing the Flesh
air_date: October 12, 2017
season_episode_number: None
overall_episode_number: None
kind: television episode
</item 14>

<item 15>
List of Scandal episodes
07 28 21 `` Any Questions ? '' Mark Tinker Matt Byrne May 9 , 2013 ( 2013 - 05 - 09 ) 221 8.87 29 22 `` White Hat 's Back On '' Tom Verica Shonda Rhimes May 16
, 2013 ( 2013 - 05 - 16 ) 222 9.12 Season 3 ( 2013 -- 14 ) ( edit ) Main article : Scandal ( season 3 ) No . overall No. in season Title Directed by Written by
Original air date Prod . code U.S. viewers ( millions ) 30 `` It 's Handled '' Tom Verica Shonda Rhimes October 3 , 2013 ( 2013 - 10 - 03 ) 301 10.52 31 ``
Guess Who 's Coming to Dinner '' Allison Liddi - Brown Heather Mitchell October 10 , 2013 ( 2013 - 10 - 10 ) 302 9.01 32 `` Mrs. Smith Goes to Washington ''
Jeannot Szwarc Matt Byrne October 17 ,
</item 15>

<item 16>
List of Scandal episodes
May 26 , 2016 . Retrieved May 27 , 2016 . Jump up ^ `` Final 2016 - 17 TV Rankings : ' Sunday Night Football ' Winning Streak Continues '' . Deadline Hollywood
. May 26 , 2017 . Retrieved May 26 , 2017 . Jump up ^ Ausiello , Michael ( June 26 , 2017 ) . `` Scandal Farewell Update : Final Season Episode Count Revealed
'' . TVLine . Retrieved July 28 , 2017 . Jump up ^ Gelman , Vlada ( January 8 , 2018 ) . `` Quantico Season 3 and Shondaland 's For the People Get ABC Premiere
Dates '' . TVLine . Retrieved January 8 , 2018 . Jump up ^ Bibel , Sara ( April 6 , 2012 ) . `` Thursday Final Ratings : ' Big Bang Theory ' , ' American Idol '
, ' Person of Interest ' , ' Missing ' , ' Up All Night ' Adjusted Up ; ' Scandal ' Adjusted Down '' . TV by the Numbers .
</item 16>

<item 17>
List of Scandal episodes
List of Scandal episodes - wikipedia List of Scandal episodes Jump to : navigation , search Scandal is an American political thriller television series created
by Shonda Rhimes , which premiered on ABC on April 5 , 2012 . Kerry Washington stars as Olivia Pope , a former White House Communications Director who leaves to
start her own crisis management firm , Pope and Associates , where she works to keep the secrets and protect the public images of the country 's most powerful
and elite . On staff at the new firm are fresh - faced lawyer Quinn Perkins ( Katie Lowes ) ; smooth talking litigator Harrison Wright ( Columbus Short ) ;
investigator Abby Whelan ( Darby Stanchfield ) ; and computer hacker with a CIA past Huck ( Guillermo Diaz ) . The show also focuses on President of the United
States Fitzgerald Grant ( Tony Goldwyn ) ; First Lady Mellie Grant ( Bellamy Young ) ; White House Chief of Staff Cyrus Beene ( Jeff Perry ) ; Vice President
and eventual presidential candidate Sally Langston ( Kate Burton ) ; U.S.
</item 17>

<item 18>
Graph query row:
series: Scandal
season: Season 7
episode: Watch Me
air_date: October 5, 2017
season_episode_number: None
overall_episode_number: None
kind: television episode
</item 18>
=== END ITEM id=863855e554d8b6f2 ===

=== ITEM id=f05e855fda8c9600 ===
Gold sentences and facts:
Sentence 0: The Philadelphia Eagles are a professional American football franchise based in Philadelphia , Pennsylvania . The Eagles compete in the National Football League ( NFL ) as a member club of the league 's National Football Conference ( NFC ) East division . They are Super Bowl champions , having won Super Bowl LII , their fourth NFL title , after winning in 1948 , 1949 , and 1960 .
  Fact 0.0: The Philadelphia Eagles are a professional American football franchise.
  Fact 0.1: The Philadelphia Eagles are based in Philadelphia, Pennsylvania.
  Fact 0.2: The Philadelphia Eagles compete in the National Football League (NFL).
  Fact 0.3: The Philadelphia Eagles are a member club of the NFL's National Football Conference (NFC) East division.
  Fact 0.4: The Philadelphia Eagles are Super Bowl champions.
  Fact 0.5: The Philadelphia Eagles won Super Bowl LII.
  Fact 0.6: Super Bowl LII was the Eagles' fourth NFL title.
  Fact 0.7: The Philadelphia Eagles won NFL titles in 1948, 1949, and 1960.
Sentence 1: Philadelphia Eagles Current season Established July 8 , 1933 ; 84 years ago ( July 8 , 1933 ) First season : 1933 Play in Lincoln Financial Field Philadelphia , Pennsylvania Headquartered in the NovaCare Complex Philadelphia , Pennsylvania Logo Wordmark League / conference affiliations National Football League ( 1933 -- present ) Eastern Division ( 1933 -- 1949 ) American Conference ( 1950 -- 1952 ) Eastern Conference ( 1953 -- 1969 ) Capitol Division ( 1967 -- 1969 ) National Football Conference ( 1970 -- present ) NFC East ( 1970 -- present ) Current uniform Team colors Midnight green , Silver , Black , White Fight song Fly , Eagles Fly Mascot Swoop Personnel Owner ( s ) Jeffrey Lurie Christina Weiss Lurie Chairman Jeffrey Lurie CEO Jeffrey Lurie President Don Smolenski General manager Howie Roseman Head coach Doug Pederson Team history Philadelphia Eagles ( 1933 -- 1942 ; 1944 -- present ) Phil - Pitt `` Steagles '' ( 1943 ) Team nicknames The Birds Championships League championships ( 4 ) NFL championships ( pre-1970 AFL -- NFL merger ) ( 3 ) 1948 , 1949 , 1960 Super Bowl championships ( 1 ) 2017 ( LII ) Conference championships ( 4 ) NFL Eastern : 1960 NFC : 1980 , 2004 , 2017 Division championships ( 13 ) NFL East : 1947 , 1948 , 1949 NFC East : 1980 , 1988 , 2001 , 2002 , 2003 , 2004 , 2006 , 2010 , 2013 , 2017 Playoff appearances ( 25 ) NFL : 1947 , 1948 , 1949 , 1960 , 1978 , 1979 , 1980 , 1981 , 1988 , 1989 , 1990 , 1992 , 1995 , 1996 , 2000 , 2001 , 2002 , 2003 , 2004 , 2006 , 2008 , 2009 , 2010 , 2013 , 2017 Home fields Baker Bowl ( 1933 -- 1935 ) Philadelphia Municipal Stadium ( 1936 -- 1939 , 1941 ) Connie Mack Stadium ( 1940 , 1942 -- 1957 ) Franklin Field ( 1958 -- 1970 ) Veterans Stadium ( 1971 -- 2002 ) Lincoln Financial Field ( 2003 -- present )
  Fact 1.0: The Philadelphia Eagles were established July 8, 1933, 84 years ago.
  Fact 1.1: The Philadelphia Eagles' first season was 1933.
  Fact 1.2: The Philadelphia Eagles play in Lincoln Financial Field in Philadelphia, Pennsylvania.
  Fact 1.3: The Philadelphia Eagles are headquartered in the NovaCare Complex in Philadelphia, Pennsylvania.
  Fact 1.4: The Philadelphia Eagles' team colors are midnight green, silver, black and white.
  Fact 1.5: The Philadelphia Eagles' fight song is 'Fly, Eagles Fly'.
  Fact 1.6: The Philadelphia Eagles' mascot is Swoop.
  Fact 1.7: The Philadelphia Eagles are owned by Jeffrey Lurie and Christina Weiss Lurie.
  Fact 1.8: Jeffrey Lurie is the chairman of the Philadelphia Eagles.
  Fact 1.9: Jeffrey Lurie is the CEO of the Philadelphia Eagles.
  Fact 1.10: Don Smolenski is the president of the Philadelphia Eagles.
  Fact 1.11: Howie Roseman is the general manager of the Philadelphia Eagles.
  Fact 1.12: Doug Pederson is the head coach of the Philadelphia Eagles.
  Fact 1.13: The Philadelphia Eagles' team history includes the Philadelphia Eagles from 1933 to 1942 and from 1944 to the present.
  Fact 1.14: The Philadelphia Eagles' team history includes the Phil-Pitt 'Steagles' in 1943.
  Fact 1.15: The Philadelphia Eagles are nicknamed The Birds.
  Fact 1.16: The Philadelphia Eagles have won 4 league championships.
  Fact 1.17: The Philadelphia Eagles have won 3 NFL championships before the AFL-NFL merger, in 1948, 1949, and 1960.
  Fact 1.18: The Philadelphia Eagles have won 1 Super Bowl championship, in 2017 (LII).
  Fact 1.19: The Philadelphia Eagles have won 4 conference championships.
  Fact 1.20: The Philadelphia Eagles won the NFL Eastern conference championship in 1960.
  Fact 1.21: The Philadelphia Eagles won the NFC conference championship in 1980, 2004 and 2017.
  Fact 1.22: The Philadelphia Eagles have won 13 division championships.
  Fact 1.23: The Philadelphia Eagles won NFL East division championships in 1947, 1948 and 1949.
  Fact 1.24: The Philadelphia Eagles won NFC East division championships in 1980, 1988, 2001, 2002, 2003, 2004, 2006, 2010, 2013 and 2017.
  Fact 1.25: The Philadelphia Eagles have made 25 playoff appearances in the NFL.
  Fact 1.26: The Philadelphia Eagles' NFL playoff appearances were in 1947, 1948, 1949, 1960, 1978, 1979, 1980, 1981, 1988, 1989, 1990, 1992, 1995, 1996, 2000, 2001, 2002, 2003, 2004, 2006, 2008, 2009, 2010, 2013 and 2017.
  Fact 1.27: The Philadelphia Eagles played home games at Baker Bowl from 1933 to 1935.
  Fact 1.28: The Philadelphia Eagles played home games at Philadelphia Municipal Stadium in 1936 to 1939 and 1941.
  Fact 1.29: The Philadelphia Eagles played home games at Connie Mack Stadium in 1940 and 1942 to 1957.
  Fact 1.30: The Philadelphia Eagles played home games at Franklin Field from 1958 to 1970.
  Fact 1.31: The Philadelphia Eagles played home games at Veterans Stadium from 1971 to 2002.
  Fact 1.32: The Philadelphia Eagles have played home games at Lincoln Financial Field since 2003.
  Fact 1.33: The Philadelphia Eagles have been affiliated with the National Football League since 1933.
  Fact 1.34: The Philadelphia Eagles were in the Eastern Division from 1933 to 1949.
  Fact 1.35: The Philadelphia Eagles were in the American Conference from 1950 to 1952.
  Fact 1.36: The Philadelphia Eagles were in the Eastern Conference from 1953 to 1969.
  Fact 1.37: The Philadelphia Eagles were in the Capitol Division from 1967 to 1969.
  Fact 1.38: The Philadelphia Eagles have been in the National Football Conference since 1970.
  Fact 1.39: The Philadelphia Eagles have been in the NFC East since 1970.

Retrieved items:
<item 1>
Philadelphia Eagles
Season records Main article : List of Philadelphia Eagles seasons Regular season record ( all - time ) : 568 -- 594 -- 26 Playoff record ( all - time ) : 22 --
21 ( last appearance after beating New England Patriots in Super Bowl LII ) Most points in a season : 474 points ( 2014 ) NFL championships won : 4 ( 3 before
the 1967 NFL - AFL merger created the Super Bowl ) Super Bowls won : 1 out of 3 appearances Passing leader ( all - time ) : Donovan McNabb -- 32,873 yards
Rushing leader ( all - time ) : LeSean McCoy -- 6,792 yards Receiving leader ( all - time ) : Harold Carmichael -- 8,978 yards Winningest coach ( all - time ) :
Andy Reid -- 130 wins Top player by approximate value ( all - time ) : Donovan McNabb -- 126 AV Rivalries New York Giants Main article : Eagles -- Giants
rivalry One of the NFL 's oldest , this rivalry began on October 15 , 1933 when the Giants defeated the newly founded Eagles 56 -- 0 .
</item 1>

<item 2>
Philadelphia Eagles
This would be Philadelphia 's first Super Bowl win and their fourth league championship ( their first championship since 1960 ) . Foles won Super Bowl MVP going
28 -- 43 with 373 passing yards , 3 touchdowns , and an interception . Nick Foles became the first backup quarterback to start and win a Super Bowl since his
opponent Tom Brady won as the backup for Drew Bledsoe in 2001 .
</item 2>

<item 3>
Philadelphia Eagles
As a result , the Eagles and Steelers no longer played each other every year . Currently they are scheduled to meet once every four years in the regular season
, the most recent meeting being in 2016 at Philadelphia , with the Eagles winning 34 - 3 . The Steelers have lost nine straight games on the road against the
Eagles dating back to 1966 , which was also the start of the Super Bowl era . The Eagles lead the all - time series 47 -- 28 -- 3 . Logo and uniforms See also :
Uniform ( American football ) and footnote about Eagles ' uniform numbers . For several decades , the Eagles ' colors were kelly green , silver , and white . In
1954 , the Eagles , along with the Baltimore Colts , became the second team ever in the NFL to put a logo on their helmets , with silver wings on a kelly green
helmet . In 1969 , the team wore two helmet versions : Kelly green with white wings in road games , and white with kelly green wings at home . From 1970 to ' 73
, they wore the white helmets with Kelly green wings exclusively before switching back to Kelly green helmets with silver wings .
</item 3>

<item 4>
Philadelphia Eagles
Foles had his best game since week 15 and threw for 352 passing yards and 3 touchdowns including an impressive flea flicker touchdown pass . This win marked the
franchise 's third Super Bowl appearance and a berth in Super Bowl LII for a 2005 rematch against Tom Brady and the New England Patriots . With Foles at the
helm , the Eagles started off the game strong , leading the Patriots 22 -- 12 at halftime . In the second half the Patriots gained momentum and for the first
time took the lead in the 4th quarter 33 -- 32 . The Eagles rallied back and scored an 11 - yard Touchdown to tight end Zach Ertz . Later the Eagles converted
on a 46 - yard field goal by Jake Elliott to lead 41 -- 33 . On the final play of the game with 9 seconds left , Brady threw a hail - mary pass directed toward
tight end Rob Gronkowski which was bobbled by Patrick Robinson before being dropped in the end - zone as time expired , securing the win for the Eagles 41 -- 33
, and ending the 58 - year long championship drought .
</item 4>

<item 5>
Philadelphia Eagles
As of the conclusion of the 2016 season , during the Lurie era , the Eagles are 1 -- 4 in conference championship games and 0 -- 1 in Super Bowls . On August 13
, 2009 , the Eagles signed quarterback Michael Vick . On December 6 , 2009 , Andy Reid became only the 5th coach in NFL history to win 100 or more games with a
single team in a single decade . The other four are Tom Landry , Don Shula , Tony Dungy , and Bill Belichick , all Super Bowl winners . McNabb finally had a
complete receiving corps , between first round draft pick Jeremy Maclin , DeSean Jackson 's 1,000 yard season , and Brent Celek ranking among the top 5 tight
ends in the league . Without Brian Dawkins , defensive end Trent Cole stepped up and became the dominant force on defense with 12 sacks , earning him his second
trip to the Pro Bowl and All - Pro honors .
</item 5>

<item 6>
Philadelphia Eagles
1944 -- present ) Phil - Pitt `` Steagles '' ( 1943 ) Team nicknames The Birds Championships League championships ( 4 ) NFL championships ( pre-1970 AFL -- NFL
merger ) ( 3 ) 1948 , 1949 , 1960 Super Bowl championships ( 1 ) 2017 ( LII ) Conference championships ( 4 ) NFL Eastern : 1960 NFC : 1980 , 2004 , 2017
Division championships ( 13 ) NFL East : 1947 , 1948 , 1949 NFC East : 1980 , 1988 , 2001 , 2002 , 2003 , 2004 , 2006 , 2010 , 2013 , 2017 Playoff appearances (
25 ) NFL : 1947 , 1948 , 1949 , 1960 , 1978 , 1979 , 1980 , 1981 , 1988 , 1989 , 1990 , 1992 , 1995 , 1996 , 2000 , 2001 , 2002 , 2003 , 2004 , 2006 , 2008 ,
</item 6>

<item 7>
Graph query row:
organization: Philadelphia Eagles
relationship_type: PARTICIPATED_IN
connection: None
relation: None
relationship: None
fact: None
event: Super Bowl LII
game_date: None
</item 7>

<item 8>
Philadelphia Eagles
Philadelphia Eagles - Wikipedia Philadelphia Eagles Jump to : navigation , search Philadelphia Eagles Current season Established July 8 , 1933 ; 84 years ago (
July 8 , 1933 ) First season : 1933 Play in Lincoln Financial Field Philadelphia , Pennsylvania Headquartered in the NovaCare Complex Philadelphia ,
Pennsylvania Logo Wordmark League / conference affiliations National Football League ( 1933 -- present ) Eastern Division ( 1933 -- 1949 ) American Conference (
1950 -- 1952 ) Eastern Conference ( 1953 -- 1969 ) Capitol Division ( 1967 -- 1969 ) National Football Conference ( 1970 -- present ) NFC East ( 1970 -- present
) Current uniform Team colors Midnight green , Silver , Black , White Fight song Fly , Eagles Fly Mascot Swoop Personnel Owner ( s ) Jeffrey Lurie Christina
Weiss Lurie Chairman Jeffrey Lurie CEO Jeffrey Lurie President Don Smolenski General manager Howie Roseman Head coach Doug Pederson Team history Philadelphia
Eagles ( 1933 -- 1942 ;
</item 8>

<item 9>
Philadelphia Eagles
In 2003 , they won the NFC first seed , but Westbrook went down in Week 17 , culminating in a loss to the Carolina Panthers in their 3rd straight NFC
Championship . 2004 season and Super Bowl XXXIX In 2004 , the Philadelphia Eagles had their best season since 1960 , going 13 -- 1 before resting their starters
and losing their next 2 , clinching the 1st seed for the second year in a row . McNabb set career highs , completing 64 % of his passes for 3,875 yards , though
he did n't play all 16 games . McNabb became the first quarterback ever to throw more than 30 touchdowns and fewer than 10 interceptions in a season . His
success could be attributed to the fact that he had a reliable receiver , Terrell Owens , who got 1,200 yards and 14 touchdowns in 14 games . After beating the
Vikings and Falcons the Eagles advanced to Super Bowl XXXIX , where they dueled the New England Patriots .
</item 9>

<item 10>
Philadelphia Eagles
External links Wikimedia Commons has media related to Philadelphia Eagles . Official website Philadelphia Eagles Founded in 1933 Based and headquartered in
Philadelphia , Pennsylvania Franchise Franchise History Seasons Coaches Quarterbacks Players Stadiums Baker Bowl Philadelphia Municipal Stadium Connie Mack
Stadium Franklin Field Veterans Stadium Lincoln Financial Field Culture Fly , Eagles Fly Swoop Curse of Billy Penn The Lombardi Curse Invincible Philadelphia
Sports Hall of Fame Matt Guokas Sr . Dan Baker Cheerleaders Silver Linings Playbook The Garbage Picking Field Goal Kicking Philadelphia Phenomenon Lore
Frankford Yellow Jackets Pennsylvania Keystoners `` Pennsylvania Polka '' Steagles `` Happy Hundred '' Miracle at the Meadowlands Buddy Ball Fog Bowl Bounty
Bowl series Body Bag Game 4th and 26 Miracle at the New Meadowlands Philly Special Rivalries Dallas Cowboys New York Giants Washington Redskins Pittsburgh
Steelers Division championships ( 13 ) 1947 1948 1949 1980 1988 2001 2002 2003 2006 2013 2017 Conference championships ( 4 ) 1960 1980 2017 League championships
( 4 ) 1948 1949 1960 2017 ( LII ) Retired numbers 5
</item 10>

<item 11>
Philadelphia Eagles
2002 , 2003 , 2004 , 2006 , 2008 , 2009 , 2010 , 2013 , 2017 Home fields Baker Bowl ( 1933 -- 1935 ) Philadelphia Municipal Stadium ( 1936 -- 1939 , 1941 )
Connie Mack Stadium ( 1940 , 1942 -- 1957 ) Franklin Field ( 1958 -- 1970 ) Veterans Stadium ( 1971 -- 2002 ) Lincoln Financial Field ( 2003 -- present ) The
Philadelphia Eagles are a professional American football franchise based in Philadelphia , Pennsylvania . The Eagles compete in the National Football League (
NFL ) as a member club of the league 's National Football Conference ( NFC ) East division . They are Super Bowl champions , having won Super Bowl LII , their
fourth NFL title , after winning in 1948 , 1949 , and 1960 . The franchise was established in 1933 as a replacement for the bankrupt Frankford Yellow Jackets ,
when a group led by Bert Bell secured the rights to an NFL franchise in Philadelphia .
</item 11>
=== END ITEM id=f05e855fda8c9600 ===

=== ITEM id=d87209eb6bcdc1b8 ===
Gold sentences and facts:
Sentence 0: `` Like a Boss '' is a song written and recorded by comedy hip hop troupe The Lonely Island for their debut studio album Incredibad . The song is a parody of the first official single from Slim Thug 's first album Already Platinum , also named `` Like a Boss '' . The song had multiple versions , including an audio - only that did not have the celebrity interviewer and a clean version that was produced in both audio and video . In 2011 it was also subject to a lawsuit from the producers of the song who claimed they never received proper compensation for their work .
  Fact 0.0: 'Like a Boss' is a song written and recorded by the comedy hip hop troupe The Lonely Island.
  Fact 0.1: 'Like a Boss' was recorded for The Lonely Island's debut studio album Incredibad.
  Fact 0.2: 'Like a Boss' is a parody of the first official single from Slim Thug's first album Already Platinum.
  Fact 0.3: The first official single from Slim Thug's first album Already Platinum is also named 'Like a Boss'.
  Fact 0.4: The song 'Like a Boss' had multiple versions.
  Fact 0.5: One version of 'Like a Boss' was audio-only and did not have the celebrity interviewer.
  Fact 0.6: A clean version of 'Like a Boss' was produced in both audio and video.
  Fact 0.7: In 2011, 'Like a Boss' was subject to a lawsuit.
  Fact 0.8: The lawsuit over 'Like a Boss' was from the producers of the song.
  Fact 0.9: The producers claimed they never received proper compensation for their work.

Retrieved items:
<item 1>
Graph query row:
song: Like a Boss
connected_entity: Slim Thug
connection: ['FROM_CHUNK', 'NEXT_CHUNK', 'FROM_CHUNK']
creator: A. Banks
</item 1>

<item 2>
Like a Boss
In 2011 it was also subject to a lawsuit from the producers of the song who claimed they never received proper compensation for their work . Contents ( hide ) 1
Song 2 Video 3 Popularity 4 Lawsuit 5 Censorship 6 References 7 External links Song ( edit ) `` Like a Boss '' premiered as a Saturday Night Live Digital Short
on April 4 , 2009 . In the song , Andy Samberg is a businessman , evidently in middle management , having a performance review . The reviewer asks him to
describe the things he does in an average day , and does so for the remainder of the song in his trademark aggressive style while repeating the lyric `` like a
boss '' after every activity . He first lists activities which are fairly mundane , such as communicating to bigger bosses in corporate headquarters , approving
memorandums , remembering birthdays , leading a workshop , and promoting synergy . However , the song takes a foul turn after he mentions his female subordinate
Deborah , whom he repeatedly pursues , rejecting him , causing him to `` swallow sadness '' , face a sexual harassment lawsuit and be passed up for a promotion
.
</item 2>

<item 3>
Like a Boss
TV Guide.com . Retrieved 23 June 2014 . Jump up ^ Kaufman , Gill . `` ' SNL , ' Andy Samberg Sued By Producers over ' Like A Boss ' - MTV '' . MTV News .
Retrieved 11 July 2014 . External links ( edit ) Beard , Lanford ( November 8 , 2011 ) . `` NBC , Andy Samberg sued for copyright infringement '' .
Entertainment Weekly . Retrieved November 8 , 2011 . Original video : `` Like a Boss '' on YouTube Censored video : `` Like a Boss - Clean Version '' on YouTube
Video and Song Lyrics : `` Like a Boss '' Lonely Island 's Song Page : `` Like a Boss '' ( hide ) The Lonely Island Andy Samberg Akiva Schaffer Jorma Taccone
Discography Studio albums Incredibad Turtleneck & Chain The Wack Album Popstar : Never Stop Never Stopping Singles `` Dick in a Box '' `` Jizz in My Pants '' ``
I 'm on a Boat '' `` I Just Had Sex '' `` The Creep '' `` We 're Back ! ''
</item 3>

<item 4>
Like a Boss
Like a Boss - wikipedia Like a Boss Jump to : navigation , search ( hide ) This article has multiple issues . Please help improve it or discuss these issues on
the talk page . ( Learn how and when to remove these template messages ) This article 's lead section does not adequately summarize key points of its contents .
Please consider expanding the lead to provide an accessible overview of all important aspects of the article . Please discuss this issue on the article 's talk
page . ( January 2015 ) This article needs additional citations for verification . Please help improve this article by adding citations to reliable sources .
Unsourced material may be challenged and removed .
</item 4>

<item 5>
Like a Boss
`` ' SNL ' Digital Short : ' Like A Boss ' ( With Video ) '' . The Ripcord . Archived from the original on 11 April 2009 . Retrieved 2014 - 06 - 21 . Jump up ^
`` Google Trends - websearch interest : like a boss - Worldwide , 2004 - present '' . Google Trends . Jump up ^ `` Twitter / search - # likeaboss '' .
Twitter.com . Jump up ^ `` like a boss on Tumblr '' . Tumblr.com . Jump up ^ `` # likeaboss '' . Facebook.com . Jump up ^ `` Like a BUS ! Playlist '' .
Youtube.com . Jump up ^ `` Like A Boss : SNL Digital Short ( VIDEO ) '' . The Huffington Post . 6 May 2009 . Retrieved 2014 - 06 - 23 . Jump up ^ DiNunno , Gina
. `` Today 's News : Our Take - Seth Rogen and Andy Samberg in SNL Digital Short `` Like a Boss '' `` . TV Guide.com . Retrieved 23 June 2014 .
</item 5>

<item 6>
Graph query row:
song: Like a Boss
connected_entity: The Lonely Island
connection: ['FROM_CHUNK', 'NEXT_CHUNK', 'FROM_CHUNK']
creator: Slim Thug
</item 6>

<item 7>
Graph query row:
song: Like a Boss
connected_entity: Slim Thug
connection: ['FROM_CHUNK', 'NEXT_CHUNK', 'FROM_CHUNK']
creator: Andy Samberg
</item 7>

<item 8>
Like a Boss
Abby Elliott and Bobby Moynihan can be seen in the video as background dancers . Popularity ( edit ) Prior to the release of the video on Saturday Night Live ,
the term `` Like a Boss '' was not often searched for or used in social context . Though it was indeed searched for prior to the release of the video , the term
clearly spiked at release and peaked on Google in April 2009 , hitting a 100 on the Google Trend chart . It has also become popular on social media sites such
as Twitter Tumblr , and Facebook , where it is a popular hashtag to cite a task completed , goal achieved or successful action that someone is proud of doing .
It has also led to parody videos such as `` Like a Bus '' , where a bus is shown flying through traffic and often bad weather conditions with no regard or
slowing for the bad conditions . On 22 October 2012 , `` Like A Boss '' surpassed 100 million views . The video was featured by The Huffington Post , and TV
Guide . Lawsuit ( edit ) In November 2011 a production team from St. Louis filed charges against Andy Samberg and Co. , NBC Universal and Universal Republic
Records .
</item 8>

<item 9>
Like a Boss
Banks , however , claims that he has yet to see any sort of a royalty for the song . Another track from the duo was later featured on the band 's 2011 album
Turtleneck & Chain in the song `` Shy Ronnie 2 : Ronnie & Clyde . '' Censorship ( edit ) This section does not cite any sources . Please help improve this
section by adding citations to reliable sources . Unsourced material may be challenged and removed . ( January 2015 ) ( Learn how and when to remove this
template message ) On the NBC version of this digital short , the part where Andy has a gun in his mouth and whines , `` Oh fuck , man , I ca n't fuckin ' do it
. Shit ! '' had the shot of Andy with the gun in his mouth shortened and redubbed the line to , `` Oh crap , man , I ca n't effin ' do it . Ugh ! ! ''
References ( edit ) Jump up ^ Van Dril , Eric ( 5 April 2009 ) . `` ' SNL ' Digital Short : ' Like A Boss ' ( With Video ) '' .
</item 9>

<item 10>
Graph query row:
song: Like a Boss
connected_entity: Slim Thug
connection: ['FROM_CHUNK', 'NEXT_CHUNK', 'FROM_CHUNK']
creator: Akiva Schaffer
</item 10>

<item 11>
Graph query row:
song: Like a Boss
connected_entity: Slim Thug
connection: ['FROM_CHUNK', 'NEXT_CHUNK', 'FROM_CHUNK']
creator: The Lonely Island
</item 11>

<item 12>
Graph query row:
song: Like a Boss
connected_entity: The Lonely Island
connection: ['FROM_CHUNK', 'NEXT_CHUNK', 'FROM_CHUNK']
creator: A. Banks
</item 12>

<item 13>
Like a Boss
The video ends with the giant fish delivering the final words `` like a boss . '' Video ( edit ) The video features Seth Rogen , recording over the vocals done
by Akiva Schaffer on the single. The video adds additional visuals not included with the words of the song . It ends with Samberg 's tombstone being shown (
which has the epitaph `` He was da Best '' ) , and a skeptical Rogen trying to confirm some of the more bizarre events with Samberg . The `` bomb the Russians
'' part shows Samberg attacking St. Basil 's Cathedral . The video features brief cameo appearances from Saturday Night Live writer John Mulaney as the worker
that Samberg micromanages , Casey Wilson as the woman giving Samberg the memo , Kristen Wiig as Samberg 's office crush Debra , Jason Sudeikis as Samberg 's
boss and Bill Hader as the man on whom Samberg performs fellatio . It also features Jorma Taccone as the man giving him the lawsuit , and Akiva Schaffer selling
him the gun . Abby Elliott and Bobby Moynihan can be seen in the video as background dancers .
</item 13>

<item 14>
Like a Boss
Unsourced material may be challenged and removed . ( February 2011 ) ( Learn how and when to remove this template message ) ( Learn how and when to remove this
template message ) `` Like a Boss '' Song by The Lonely Island featuring Seth Rogen from the album Incredibad Recorded 2008 Genre Comedy hip hop Length 1 : 46
Label Universal Republic Records Songwriter ( s ) Andy Samberg , Akiva Schaffer , Jorma Taccone , A. Banks Producer ( s ) Rick tha Rular Incredibad track
listing `` Shrooms ( skit ) '' ( 9 ) `` Like a Boss '' ( 10 ) `` We Like Sportz '' ( 11 ) `` Like a Boss '' is a song written and recorded by comedy hip hop
troupe The Lonely Island for their debut studio album Incredibad . The song is a parody of the first official single from Slim Thug 's first album Already
Platinum , also named `` Like a Boss '' . The song had multiple versions , including an audio - only that did not have the celebrity interviewer and a clean
version that was produced in both audio and video .
</item 14>

<item 15>
Graph query row:
song: Like a Boss
connected_entity: Slim Thug
connection: ['FROM_CHUNK', 'NEXT_CHUNK', 'FROM_CHUNK']
creator: Slim Thug
</item 15>

<item 16>
Graph query row:
song: Like a Boss
connected_entity: The Lonely Island
connection: ['FROM_CHUNK', 'NEXT_CHUNK', 'FROM_CHUNK']
creator: Akiva Schaffer
</item 16>

<item 17>
Like a Boss
The ensuing depression leads to him binge drinking vodka and defecating on Deborah 's desk . The activities Samberg 's character describes become more and more
bizarre , such as calling a sex line , `` cry ( ing ) deeply '' , demanding a refund , attempting suicide with a gun only to back out in fear , vomiting on
Deborah 's desk , jumping out of a window , `` suck ( ing ) a dude 's dick '' , purchasing cocaine , crashing his car , `` suck ( ing ) his own dick , '' eating
chicken strips , castrating himself , blacking out in a sewer , meeting a giant fish which he evidently `` fucks its brains out '' , turning into a jet , ``
bombing the Russians '' , crashing into the sun , and finally dying . Everything described is done `` like a boss . '' It ends with the reviewer trying to
confirm some of the more bizarre events with Samberg , who confirms that he does daily chop off his testicles and die , but denies ever having said he fellates
himself . While Samberg then keeps repeating the words `` I 'm the boss , '' the reviewer leaves the room , understandably annoyed and disgusted .
</item 17>

<item 18>
Like a Boss
`` Jack Sparrow '' `` 3 - Way ( The Golden Rule ) '' `` YOLO '' `` Spring Break Anthem '' `` Hugs '' Other songs `` Lazy Sunday '' `` Like a Boss '' `` Cool
Guys Do n't Look at Explosions '' `` Everything Is Awesome '' Film Hot Rod ( 2007 ) Popstar : Never Stop Never Stopping ( 2016 ) Television SNL Digital Shorts
Party Over Here Book Retrieved from `` https://en.wikipedia.org/w/index.php?title=Like_a_Boss&oldid=816720124 '' Categories : The Lonely Island songs 2009
singles Saturday Night Live sketches Comedy rap songs Hidden categories : Wikipedia introduction cleanup from January 2015 All pages needing cleanup Articles
covered by WikiProject Wikify from January 2015 All articles covered by WikiProject Wikify Articles needing additional references from February 2011 All
articles needing additional references Articles with multiple maintenance issues Articles with hAudio microformats Articles needing additional references from
January 2015 Talk Contents About Wikipedia Nederlands Simple English Edit links This page was last edited on 23 December 2017 , at 06 : 20 . Text is available
under the Creative Commons Attribution - ShareAlike License ; additional terms may apply .
</item 18>

<item 19>
Graph query row:
song: Like a Boss
connected_entity: The Lonely Island
connection: ['FROM_CHUNK', 'NEXT_CHUNK', 'FROM_CHUNK']
creator: Jorma Taccone
</item 19>

<item 20>
Graph query row:
song: Like a Boss
connected_entity: Slim Thug
connection: ['FROM_CHUNK', 'NEXT_CHUNK', 'FROM_CHUNK']
creator: Jorma Taccone
</item 20>
=== END ITEM id=d87209eb6bcdc1b8 ===

=== ITEM id=57f81472fc8172b6 ===
Gold sentences and facts:
Sentence 0: The Whistleblower Protection Act of 1989 , Pub. L. 101 - 12 as amended , is a United States federal law that protects federal whistleblowers who work for the government and report agency misconduct . A federal agency violates the Whistleblower Protection Act if agency authorities take ( or threaten to take ) retaliatory personnel action against any employee or applicant because of disclosure of information by that employee or applicant . Whistleblowers may file complaints that they believe reasonably evidences a violation of a law , rule or regulation ; gross mismanagement ; gross waste of funds ; an abuse of authority ; or a substantial and specific danger to public health or safety .
  Fact 0.0: The Whistleblower Protection Act of 1989 is Pub. L. 101-12, as amended.
  Fact 0.1: The Whistleblower Protection Act of 1989 is a United States federal law.
  Fact 0.2: The Whistleblower Protection Act of 1989 protects federal whistleblowers who work for the government and report agency misconduct.
  Fact 0.3: A federal agency violates the Whistleblower Protection Act if agency authorities take or threaten to take retaliatory personnel action against any employee or applicant.
  Fact 0.4: A federal agency violates the Whistleblower Protection Act if the retaliatory personnel action is because of disclosure of information by that employee or applicant.
  Fact 0.5: Whistleblowers may file complaints that they believe reasonably evidence a violation of a law, rule or regulation.
  Fact 0.6: Whistleblowers may file complaints about gross mismanagement.
  Fact 0.7: Whistleblowers may file complaints about gross waste of funds.
  Fact 0.8: Whistleblowers may file complaints about an abuse of authority.
  Fact 0.9: Whistleblowers may file complaints about a substantial and specific danger to public health or safety.

Retrieved items:
<item 1>
Whistleblower Protection Act
The most common characteristics for a court claim that are encompassed within the protection of the Act include : that the plaintiff is an employee or person
covered under the specific statutory or common law relied upon for action , that the defendant is an employer or person covered under the specific statutory or
common law relied upon for the action , that the plaintiff engaged in protected whistleblower activity , that the defendant knew or had knowledge that the
plaintiff engaged in such activity , that there was retaliatory action taken against the one doing the whistleblowing and that the unfair treatment would not
have occurred if the plaintiff had n't brought to attention the activities . Robert MacLean blew the whistle on the fact that the TSA had cut its funding for
more air marshals . In 2009 MacLean , represented by the Government Accountability Project , challenged his dismissal at the Merit Systems Protection Board , on
the grounds that `` his disclosure of the text message was protected under the Whistleblower Protection Act of 1989 , because he ' reasonably believe ( d ) '
that the leaked information disclosed ' a substantial and specific danger to public health or safety ' . ''
</item 1>

<item 2>
Whistleblower Protection Act
Legislative history Introduced in the Senate as S. 20 by Carl Levin ( D -- MI ) on January 25 , 1989 Passed the Senate on March 16 , 1989 ( 97 - 0 , Roll call
vote 24 , via Senate.gov ) Passed the House on March 21 , 1989 ( Agreed voice vote ) Signed into law by President George H.W. Bush on April 10 , 1989 The
Whistleblower Protection Act of 1989 , Pub. L. 101 - 12 as amended , is a United States federal law that protects federal whistleblowers who work for the
government and report agency misconduct . A federal agency violates the Whistleblower Protection Act if agency authorities take ( or threaten to take )
retaliatory personnel action against any employee or applicant because of disclosure of information by that employee or applicant . Whistleblowers may file
complaints that they believe reasonably evidences a violation of a law , rule or regulation ; gross mismanagement ; gross waste of funds ; an abuse of authority
; or a substantial and specific danger to public health or safety .
</item 2>

<item 3>
Whistleblower Protection Act
November 7 , 2014 . Retrieved April 27 , 2016 . Jump up ^ https://obamawhitehouse.archives.gov/sites/default/files/image/ppd-19.pdf Jump up ^ WHISTLEBLOWER
PROTECTION ENHANCEMENT ACT OF 2012 REPORT OF THE COMMITTEE ON HOMELAND SECURITY AND GOVERNMENTAL AFFAIRS , UNITED STATES SENATE TO ACCOMPANY S. 743 . April 19 ,
2012 Jump up ^ https://www.gpo.gov/fdsys/pkg/CRPT-112srpt155/pdf/CRPT-112srpt155.pdf Jump up ^ `` H.R. 4197 - Summary '' . United States Congress . Retrieved
July 15 , 2014 . Jump up ^ `` H.R. 4197 - All Actions '' . United States Congress . Retrieved July 15 , 2014 . Jump up ^ Joe Davidson . How FBI blocks
whistleblower fighting dismissal ; new bill could help others . Washington Post 2 Feb 2016 .
</item 3>

<item 4>
Graph query row:
act: Whistleblower Protection Act
relationship: RELATED_TO
connected_to: Whistle Blowers Protection Act, 2011
page: Whistleblower Protection Act
passage: Whistleblower Protection Act
MacLean won the case in a ruling of 7 -- 2 in the Supreme Court in January 2015 . Whistleblower Protection Enhancement Act and Presidential Policy Directive 19
( edit ) President Barack Obama issued Presidential Policy Directive 19 ( PPD - 19 ) , entitled `` Protecting Whistleblowers with Access to Classified
Information '' . According to the law signed by Obama on October 10 , 2012 , it is written that `` this Presidential Policy Directive ensures that employees ( 1
) serving in the Intelligence Community or ( 2 ) who are eligible for access to classified information can effectively report waste , fraud , and abuse while
protecting classified national security information . It prohibits retaliation against employees for reporting waste , fraud , and abuse . However , according
to a report that the Committee on Homeland Security and Governmental Affairs submitted to accompany S. 743 , `` the federal whistleblowers have seen their
protections diminish in recent years , largely as a result of a series of decisions by the United States Court of Appeals for the Federal Circuit , which has
exclusive jurisdiction over many cases brought under the Whistleblower Protection Act ( WPA ) .
</item 4>

<item 5>
Whistleblower Protection Act
4197 ; 113th Congress ) , a bill that gives authority to federal employees who want to appeal their judgment to any federal court , and which allows
whistleblowers to appeal to any U.S. Court of Appeals that has jurisdiction . The bill would extend from three years after the effective date of the
Whistleblower Protection Enhancement Act of 2012 ( i.e. , December 27 , 2012 ) , the period allowed for : ( 1 ) filing a petition for judicial review of Merit
Systems Protection Board decisions in whistleblower cases , and ( 2 ) any review of such a decision by the Director of the Office of Personnel Management ( OPM
) . There is also a bill in its early stages called the FBI Whistleblower Protection Enhancement Act . This bill was introduced by Senators Grassley and Leahy
in 2015 . It would protect whistleblowers in the FBI more completely from reprisals by supervisors , by assigning administrative law judges to adjudicate the
cases , and enabling FBI employees to appeal their decisions to the courts . This bill is still in the process of possibly getting passed and has individuals
cosponsoring it in March 2016 .
</item 5>

<item 6>
Whistleblower Protection Act
See also ( edit ) Whistleblower protection in United States False Claims Act Federal crime Immunity from prosecution Informant List of whistleblowers Qui tam
Testimony Turn state 's evidence White collar crime Witness Witness intimidation United States Federal Witness Protection Program United States Marshals Service
References ( edit ) Jump up ^ Public Law 101 - 12 ( 1989 ) at gpo.gov Jump up ^ Peter Eisler ( 2010 - 03 - 15 ) . `` Whistleblowers '' . USA TODAY . Jump up ^
High Court Trims Whistleblower Rights Jump up ^ McCarthy , Robert J. `` Blowing in the Wind : Answers for Federal Whistleblowers '' . William & Mary Policy
Review 3 : 184 ( 2012 ) . Jump up ^ `` Know Your Rights FAQ '' . National Whistleblowers Center . Retrieved April 27 , 2016 . Jump up ^ `` Whistleblower
protection case reaches US Supreme Court '' . Courage Foundation . November 7 , 2014 . Retrieved April 27 , 2016 .
</item 6>

<item 7>
Whistleblower Protection Act
Whistleblower Protection Act - Wikipedia Whistleblower Protection Act Jump to : navigation , search For the Act in India , see Whistle Blowers Protection Act ,
2011 . Whistleblower Protection Act Long title An Act to amend title 5 , United States Code , to strengthen the protections available to Federal employees
against prohibited personnel practices , and for other purposes . Nicknames Whistleblower Protection Act of 1989 Enacted by the 101st United States Congress
Effective April 10 , 1989 Citations Public law 101 - 12 Statutes at Large 103 Stat. 16 Codification Titles amended 5 U.S.C. : Government Organization and
Employees U.S.C. sections amended 5 U.S.C. ch. 12 § 1201 et seq . Legislative history Introduced in the Senate as S.
</item 7>

<item 8>
Whistleblower Protection Act
Contents ( hide ) 1 Authorized Federal Agencies 2 Legal Cases 3 Whistleblower Protection Enhancement Act and Presidential Policy Directive 19 4 Related
legislation 5 See also 6 References 7 External links Authorized federal agencies ( edit ) The Office of Special Counsel investigates federal whistleblower
complaints . In October 2008 , then - special counsel Scott Bloch resigned amid an FBI investigation into whether he obstructed justice by illegally deleting
computer files following complaints that he had retaliated against employees who disagreed with his policies . Then - Senator Barack Obama made a campaign vow
to appoint a special counsel committed to whistleblower rights . It was not until April 2011 that President Obama 's appointee Carolyn Lerner was confirmed by
the Senate . Today , the primary mission of OSC is to safeguard the merit system by protecting federal employees and applicants from prohibited personnel
practices , especially reprisal for whistleblowing . The Merit Systems Protection Board , a quasi-judicial agency that adjudicates whistleblower complaints ,
uses appointed administrative law judges who often back the government . Since 2000 , the board has ruled for whistleblowers just three times in 56 cases
decided on their merits , according to a Government Accountability Project analysis .
</item 8>

<item 9>
Whistleblower Protection Act
Specifically , the Federal Circuit has wrongly accorded a narrow definition to the type of disclosure that qualifies for whistleblower protection . Additionally
, the lack of remedies under current law for most whistleblowers in the intelligence community and for whistleblowers who face retaliation in the form of
withdrawal of the employee 's security clearance leaves unprotected those who are in a position to disclose wrongdoing that directly affects our national
security . '' S. 743 would address these problems by restoring the original congressional intent of the WPA to adequately protect whistleblowers , by
strengthening the WPA , and by creating new whistleblower protections for intelligence employees and new protections for employees whose security clearance is
withdrawn in retaliation for having made legitimate whistleblower disclosures . S. 743 ultimately became Pub. L. 112 - 199 ( S. Rep. 112 - 155 ) . Related
legislation ( edit ) This section needs to be updated . Please update this article to reflect recent events or newly available information . ( January 2017 ) On
July 14 , 2014 , the United States House of Representatives voted to pass the All Circuit Review Extension Act ( H.R.
</item 9>

<item 10>
Whistleblower Protection Act
Legal cases ( edit ) The U.S. Supreme Court , in the case of Garcetti v. Ceballos , 04 - 473 , ruled in 2006 that government employees do not have protection
from retaliation by their employers under the First Amendment of the Constitution when they speak pursuant to their official job duties . The U.S. Merit Systems
Protection Board ( MSPB ) uses agency lawyers in the place of administrative law judges to decide federal employees ' whistleblower appeals . These lawyers ,
dubbed `` attorney examiners , '' deny 98 % of whistleblower appeals ; the Board and the Federal Circuit Court of Appeals give great deference to their initial
decisions , resulting in affirmance rates of 97 % and 98 % , respectively .
</item 10>

<item 11>
Whistleblower Protection Act
Obama appointed a new chairperson and vice chairperson with backgrounds as federal worker advocates , but Tom Devine of GAP says , `` It 's likely to take years
for them to turn things around . '' Currently , this office works to protect the Merit System Principles and promote an effective Federal workforce free of
Prohibited Personnel Practices . The Court of Appeals for the Federal Circuit was established under Article III of the Constitution on October 1 , 1982 . It is
the only court empowered to hear appeals of whistleblower cases decided by the merit board , has been criticized by Senator Grassley ( R - Iowa ) and others in
Congress for misinterpreting whistleblower laws and setting a precedent that is hostile to claimants . Between 1994 and 2010 , the court had ruled for
whistleblowers in only three of 203 cases decided on their merits , GAP 's analysis found . Legal cases ( edit ) The U.S.
</item 11>
=== END ITEM id=57f81472fc8172b6 ===

=== ITEM id=0c8bc064b89776ae ===
Gold sentences and facts:
Sentence 0: The song relates to both the Old and New Testaments . The verses reflect the Israelites ' escape out of Egypt as found in Exodus 14 . The chorus refers to healing : see John 5 : 4 , `` For an angel went down at a certain season into the pool , and troubled the water : whosoever then first after the troubling of the water stepped in was made whole of whatsoever disease he had . ''
  Fact 0.0: The song relates to both the Old and New Testaments.
  Fact 0.1: The song's verses reflect the Israelites' escape out of Egypt as found in Exodus 14.
  Fact 0.2: The song's chorus refers to healing.
  Fact 0.3: The healing reference in the song's chorus is found in John 5:4.
  Fact 0.4: John 5:4 says that an angel went down at a certain season into the pool and troubled the water.
  Fact 0.5: John 5:4 says that whoever first stepped in after the troubling of the water was made whole of whatever disease he had.

Retrieved items:
<item 1>
Wade in the Water
The story follows the lives of three characters before , during , and after the great Johnstown Flood of 1889 . Selective list of recordings ( edit ) This
section does not cite any sources . Please help improve this section by adding citations to reliable sources . Unsourced material may be challenged and removed
. ( February 2017 ) ( Learn how and when to remove this template message ) The first commercially recorded version of the song was released by Paramount Records
( sung by Sunset Four Jubilee Singers in 1925 ) , as `` Good News Chariot 's Coming and Wade in the Water '' .
</item 1>

<item 2>
Wade in the Water
Encyclopedia of Great Popular Song Recordings . Rowman & Littlefield Publishers . p. 387 . ISBN 978 - 1 - 4422 - 5449 - 7 . Jump up ^ `` Wade in De Water '' .
Archived from the original on July 9 , 2008 . Retrieved January 11 , 2008 . Jump up ^ `` Archived copy '' . Archived from the original on 2008 - 07 - 24 .
Retrieved 2008 - 08 - 18 . Jump up ^ `` Pathways to Freedom Secrets : Signs and Symbols Music '' . Pathways.thinkport.org . Retrieved 2016 - 09 - 29 . Jump up ^
Robert Pruter ( 1992 ) . Chicago Soul . University of Illinois Press . p. 125 . ISBN 978 - 0 - 252 - 06259 - 9 . Jump up ^ `` Wade in the Water : A Must See
Film : Boston Globe '' . Wadeinthewater.com . Retrieved 2016 - 09 - 29 .
</item 2>

<item 3>
Wade in the Water
. Boulder , Colo. : Leave a Little Room . pp. 54 , 112 . ISBN 0976237709 . `` Music : Wade in the Water '' . Pathways to Freedom : Maryland & the Underground
Railroad Secrets : Signs and Symbols . Maryland Public Television . 2014 . Tobin , Jacqueline L. ; Dobard , Raymond G. ( 2011 ) . Hidden in plain view the
secret story of quilts and the underground railroad . New York , N.Y. : Anchor Books . p. 126 . ISBN 0307790568 .
</item 3>

<item 4>
Wade in the Water
: Anchor Books . p. 126 . ISBN 0307790568 . External links ( edit ) Lyrics at Negrospirituals.org MIDI at Gospelsonglyrics.org Lyrics of this song at
MetroLyrics Retrieved from `` https://en.wikipedia.org/w/index.php?title=Wade_in_the_Water&oldid=833167717 '' Categories : American folk songs Bob Dylan songs
African - American spiritual songs Eva Cassidy songs 1966 singles 1967 singles Year of song unknown Songs based on American history Hidden categories : Articles
needing additional references from September 2016 All articles needing additional references All articles with specifically marked weasel - worded phrases
Articles with specifically marked weasel - worded phrases from November 2015 Articles needing additional references from February 2017 Talk Contents About
Wikipedia Deutsch Italiano Edit links This page was last edited on 30 March 2018 , at 01 : 17 . Text is available under the Creative Commons Attribution -
ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark
of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 4>

<item 5>
Wade in the Water
2004 ) includes the chorus of `` Wade in the Water '' with the altered lyrics `` Straight from the Water , Straight from the water , children '' IIIrd Tyme Out
( The Best Durn Ride ) ( 2004 ) Mavis Staples ( Live : Hope at the Hideout ) ( 2008 ) J. Boykin ( Jesus Loves Me ) ( 2009 ) John Boutte , recorded for the film
Wade in the Water , Children ( see above ) Patty Griffin Downtown Church ( 2010 ) Cerys Matthews - `` Explorer '' ( 2011 ) Tedeschi Trucks Band Everybody 's
Talkin ' ( 2012 ) Aleksander Vinter sampled the first verse in a song of the same title ( 2012 ) Tina Savage The Son and the Soul ( 2013 ) Jamie N Commons
Spotify Sessions ( Jamie N Commons EP ) ( 2013 ) Frances Madden ( 2014 ) Sirena : the Sirens Out of the Blue ( 2014 ) PJ Harvey as an intro to `` River
Anacostia '' ( 2016 ) References ( edit ) ^ Jump up to : Steve Sullivan ( 17 May 2017 ) .
</item 5>

<item 6>
Wade in the Water
Peace and Happiness ( 1969 ) CCS ( 1970 ) Alex Harvey , with Rock Workshop ( 1971 ) Peter Herbolzheimer with Rhythm Combination & Brass ( 1972 ) Ellen McIlwain
Honkey Tonk Angel ( Polydor , 1972 ) Fine Young Cannibals , B - side to Blue ( 1985 ) Tony ! Toni ! Toné ! 's song `` Little Walter '' ( Who ? , 1988 ) includes
the chorus of `` Wade in the Water '' with the altered lyrics `` Hey , Little Walter ... Hey , Little Walter , listen ... Hey , Little Walter ... Somethin 's
gon na getcha , Little Walter '' Eva Cassidy ( 1997 ) Make - up Wade in the Water 7 '' ( 1997 ) Chanticleer , recorded in `` Reflections ( An Anniversary
Celebration ) '' ( 1997 ) and `` A Portrait '' ( 2005 ) Mary Mary Thankful ( 2000 ) Moses Hogan Moses Hogan Singers : Choral Series ( 2002 ) Marc Broussard 's
song `` Home '' ( Carenco ,
</item 6>

<item 7>
Wade in the Water
Wade in the Water - wikipedia Wade in the Water Jump to : navigation , search This article is about the song . For the album by Ramsey Lewis , see Wade in the
Water ( album ) . This article needs additional citations for verification . Please help improve this article by adding citations to reliable sources .
Unsourced material may be challenged and removed . ( September 2016 ) ( Learn how and when to remove this template message ) `` Wade in the Water '' American
folk song Native language ( s ) English First publication New Jubilee Songs as Sung by the Fisk Jubilee Singers ( 1901 ) Catalogue ( s ) Roud 5439 Notable
performer ( s ) The Staple Singers , Ramsey Lewis ; etc . First released recording Title of the recording `` Good News Chariot 's Coming and Wade in the Water
'' Released ca .
</item 7>

<item 8>
Wade in the Water
1925 Record label Paramount Records Artist Sunset Four Jubilee Singers Genre Negro spiritual `` Wade in the Water '' ( Roud 5439 ) is the name of a Negro
spiritual first published in New Jubilee Songs as Sung by the Fisk Jubilee Singers ( 1901 ) by John Wesley Work II and his brother , Frederick J. Work ( see
Fisk Jubilee Singers ) . It is associated with the songs of the Underground Railroad . Wade in the water Wade in the water , children Wade in the water God 's
gon na trouble the water Contents ( hide ) 1 Meaning 2 In popular culture 3 Selective list of recordings 4 References 4.1 Sources 5 External links Meaning (
edit ) The song relates to both the Old and New Testaments . The verses reflect the Israelites ' escape out of Egypt as found in Exodus 14 . The chorus refers
to healing : see John 5 : 4 , `` For an angel went down at a certain season into the pool , and troubled the water : whosoever then first after the troubling of
the water stepped in was made whole of whatsoever disease he had . ''
</item 8>

<item 9>
Wade in the Water
Many Internet sources and popular books claim that songs such as `` Wade in the Water '' contained explicit instructions to fugitive slaves on how to avoid
capture and the route to take to successfully make their way to freedom . An example of this is cited in the book Pathways to Freedom : Maryland & the
Underground Railroad . The book explains how Harriet Tubman used the song `` Wade in the Water '' to tell escaping slaves to get off the trail and into the
water to make sure that the dogs employed by slavers lost their trail . In popular culture ( edit ) One version of the song performed by The Staple Singers
became a part of the civil rights movement in the US . `` Wade in the Water '' was a popular instrumental hit in 1966 for the Ramsey Lewis Trio , which prompted
further instrumental recordings by Herb Alpert and the Tijuana Brass , and Billy Preston ( both 1967 ) . The melody was used for the 1988 Tony ! Toni ! Toné !
hit `` Little Walter '' .
</item 9>

<item 10>
Wade in the Water
Retrieved 2016 - 09 - 29 . Jump up ^ `` Wade in the Water , Children ( DVD ) Elizabeth Wood , Gabriel Nussbaum '' . IndiePix Films . Retrieved 2016 - 09 - 29 .
Jump up ^ William Edward Burghardt Du Bois ( 1925 ) . Crisis . Crisis Publishing Company . Jump up ^ Robert M. Marovich ( 15 March 2015 ) . A City Called Heaven
: Chicago and the Birth of Gospel Music . University of Illinois Press . p. 135 . ISBN 978 - 0 - 252 - 09708 - 9 . Sources ( edit ) Waltz , Robert B ; David G.
Engle . `` Wade In The Water '' . The Traditional Ballad Index : An Annotated Bibliography of the Folk Songs of the English - Speaking World . Hosted by
California State University , Fresno , Folklore , 2007 . Further reading Jones , Arthur C. ( 2005 ) . Wade in the water : the wisdom of the spirituals ( 3rd ed
. ) .
</item 10>
=== END ITEM id=0c8bc064b89776ae ===

=== ITEM id=4a074ad4d325e79c ===
Gold sentences and facts:
Sentence 0: No . Name Field Affiliation Date of Appointment Date of Retirement Roopa Ganguly Art Bharatiya Janata Party 04 - Oct - 2016 03 - Oct - 2022 Sambhaji Raje Social work Bharatiya Janata Party 07 - Jun - 2016 03 - May - 2022 Suresh Gopi Art Bharatiya Janata Party 25 - Apr - 2016 24 - Apr - 2022 Subramanian Swamy Economics Bharatiya Janata Party 25 - Apr - 2016 24 - Apr - 2022 5 Narendra Jadhav Economics Nominated 25 - Apr - 2016 24 - Apr - 2022 6 Mary Kom Sport Nominated 25 - Apr - 2016 24 - Apr - 2022 7 Swapan Dasgupta Journalism Nominated 25 - Apr - 2016 24 - Apr - 2022 8 K.T.S. Tulsi Law Nominated 25 - Feb - 2014 24 - Feb - 2020 9 K. Parasaran Law Nominated 09 - Jun - 2012 28 - Jun - 2018 10 Rekha Art Nominated 27 - Apr - 2012 26 - Apr - 2018 11 Sachin Tendulkar Social service Nominated 27 - Apr - 2012 26 - Apr - 2018 12 Anu Aga Business Nominated 27 - Apr - 2012 26 - Apr - 2018
  Fact 0.0: Roopa Ganguly, in the Art field, is affiliated with the Bharatiya Janata Party and was appointed 04-Oct-2016 and retired 03-Oct-2022.
  Fact 0.1: Sambhaji Raje, in the Social work field, is affiliated with the Bharatiya Janata Party and was appointed 07-Jun-2016 and retired 03-May-2022.
  Fact 0.2: Suresh Gopi, in the Art field, is affiliated with the Bharatiya Janata Party and was appointed 25-Apr-2016 and retired 24-Apr-2022.
  Fact 0.3: Subramanian Swamy, in the Economics field, is affiliated with the Bharatiya Janata Party and was appointed 25-Apr-2016 and retired 24-Apr-2022.
  Fact 0.4: Narendra Jadhav, in the Economics field, is nominated and was appointed 25-Apr-2016 and retired 24-Apr-2022.
  Fact 0.5: Mary Kom, in the Sport field, is nominated and was appointed 25-Apr-2016 and retired 24-Apr-2022.
  Fact 0.6: Swapan Dasgupta, in the Journalism field, is nominated and was appointed 25-Apr-2016 and retired 24-Apr-2022.
  Fact 0.7: K.T.S. Tulsi, in the Law field, is nominated and was appointed 25-Feb-2014 and retired 24-Feb-2020.
  Fact 0.8: K. Parasaran, in the Law field, is nominated and was appointed 09-Jun-2012 and retired 28-Jun-2018.
  Fact 0.9: Rekha, in the Art field, is nominated and was appointed 27-Apr-2012 and retired 26-Apr-2018.
  Fact 0.10: Sachin Tendulkar, in the Social service field, is nominated and was appointed 27-Apr-2012 and retired 26-Apr-2018.
  Fact 0.11: Anu Aga, in the Business field, is nominated and was appointed 27-Apr-2012 and retired 26-Apr-2018.

Retrieved items:
<item 1>
List of nominated members of Rajya Sabha
Dutta 1993 -- 1999 85 -- Maulana Habibur Rahman Nomani 1993 -- 1999 86 Nirmala Deshpande 1997 -- 1999 and 2004 -- 2010 87 Shabana Azmi 1997 -- 2003 88 -- P.
Selvie Das 1997 -- 2003 89 -- Kartar Singh Duggal 1997 -- 2003 90 Kuldip Nayyar 1997 -- 2003 91 -- Raja Ramanna 1997 -- 2003 92 C.
</item 1>

<item 2>
List of nominated members of Rajya Sabha
Daphtary 1972 -- 1978 47 -- Habib Tanvir 1972 -- 1978 48 -- Krishna Kriplani 1974 -- 1980 49 Lokesh Chandra 1974 -- 1986 50 -- Scato Swu 1974 -- 1986 51 -- B.N.
Banerjee 1976 -- 1982 52 -- Malcolm Adiseshiah 1978 -- 1984 53 -- Fathema Ismail 1978 -- 1984 54 -- Pandurang Dharmaji Jadhav 1978 -- 1984 55 -- Bhagwati Charan
Varma 1978 -- 1984 56 Nargis Dutt 1980 -- 1986 57 Khushwant Singh 1980 -- 1986 58 -- Asima Chatterjee 1982 -- 1990 59 Sivaji Ganesan 1982 -- 1986 60 --
Hayatullah Ansari 1982 -- 1988 61 -- Madan Bhatia 1982 -- 1994 62 -- V.N.
</item 2>

<item 3>
Graph query row:
person: Rukmini Devi Arundale
organization: Rajya Sabha
appointment_date: None
retirement_date: None
membership_period: None
term: None
start_date: None
end_date: None
tenure: 1952 -- 1962
period: None
</item 3>

<item 4>
List of nominated members of Rajya Sabha
Tiwari 1982 -- 1984 63 -- Ghulam Rasool Kar 1984 -- 1987 64 -- Tindivanam K. Ramamurthy 1984 -- 1990 65 -- H.L. Kapur 1985 -- 1988 66 Salim Ali 1985 -- 1988 67
-- Purushottam Kakodkar 1985 -- 1991 68 Ela Bhatt 1986 -- 1988 69 -- Amrita Pritam 1986 -- 1992 70 M.F. Husain 1986 -- 1992 71 R.K.
</item 4>

<item 5>
List of nominated members of Rajya Sabha
No . Name Field Affiliation Date of Appointment Date of Retirement Roopa Ganguly Art Bharatiya Janata Party 04 - Oct - 2016 03 - Oct - 2022 Sambhaji Raje Social
work Bharatiya Janata Party 07 - Jun - 2016 03 - May - 2022 Suresh Gopi Art Bharatiya Janata Party 25 - Apr - 2016 24 - Apr - 2022 Subramanian Swamy Economics
Bharatiya Janata Party 25 - Apr - 2016 24 - Apr - 2022 5 Narendra Jadhav Economics Nominated 25 - Apr - 2016 24 - Apr - 2022 6 Mary Kom Sport Nominated 25 - Apr
- 2016 24 - Apr - 2022 7 Swapan Dasgupta Journalism Nominated 25 - Apr - 2016 24 - Apr - 2022 8 K.T.S.
</item 5>

<item 6>
List of nominated members of Rajya Sabha
Kasturirangan 2003 -- 2009 102 -- Narayan Singh Manaklao 2003 -- 2009 103 -- Vidya Niwas Misra 2003 -- 2009 104 -- Chandan Mitra 2003 -- 2009 105 Dara Singh
2003 -- 2009 106 Kapila Vatsyayan 2006 -- 2012 107 Ram Jethmalani 2006 -- 2009 108 -- Shobhana Bhartia 2006 -- 2012 109 Shyam Benegal 2006 -- 2012 110 M.S.
Swaminathan 2007 -- 2013 111 -- C. Rangarajan 2008 -- 2010 112 -- H.K.
</item 6>

<item 7>
List of nominated members of Rajya Sabha
Narayana Reddy 1997 -- 2003 93 Mrinal Sen 1997 -- 2003 94 -- Chaudhary Harmohan Singh Yadav 1997 -- 2003 95 -- Nanaji Deshmukh 1999 -- 2005 96 Lata Mangeshkar
1999 -- 2005 97 -- Fali Nariman 1999 -- 2005 98 -- Cho Ramaswamy 1999 -- 2005 99 Hema Malini 2003 -- 2009 100 -- Bimal Jalan 2003 -- 2009 101 K.
</item 7>

<item 8>
List of nominated members of Rajya Sabha
Krishnamachari 1961 -- 1967 24 -- R.R. Diwakar 1962 -- 1968 25 -- Gopal Singh 1962 -- 1968 26 -- M. Ajmal Khan 1964 -- 1972 27 -- Satyavrata Siddhantalankar
1964 -- 1968 28 Shakuntala Paranjpye 1964 -- 1970 29 -- Badri Nath Prasad 1964 -- 1970 30 -- G. Ramachandran 1964 -- 1970 31 -- M.N. Kaul 1964 -- 1972 32 --
Harivanshrai Bachchan 1966 -- 1972 33 -- Dhananjay Ramchandra Gadgil 1966 -- 1972 34 -- M.C.
</item 8>

<item 9>
List of nominated members of Rajya Sabha
Setalvad 1966 -- 1972 35 G. Sankara Kurup 1968 -- 1972 36 Joachim Alva 1968 -- 1974 37 S. Nurul Hasan 1968 -- 1974 38 -- K. Ramiah 1968 -- 1974 39 -- Ganga
Sharan Sinha 1968 -- 1974 40 -- Maragatham Chandrasekhar 1970 -- 1988 41 -- Umashankar Joshi 1970 -- 1976 42 -- Rasheeduddin Khan 1970 -- 1982 43 -- Vidya
Prakash Dutt 1971 -- 1980 44 -- Abu Abraham 1972 -- 1978 45 -- Pramathanath Bishi 1972 -- 1978 46 -- C.K.
</item 9>

<item 10>
List of nominated members of Rajya Sabha
Tulsi 2014 - 2020 125 Subramanian Swamy 2016 - 2022 126 MC Mary Kom 2016 - 2022 127 Narendra Jadhav 2016 - 2022 129 Suresh Gopi 2016 - 2022 130 -- Swapan
Dasgupta 2016 - 2022 131 -- Sambhaji Raje 2016 - 2022 132 -- Roopa Ganguly 2016 - 2022 References ( edit ) Nominated Members since 1952 Narender Kumar Current
list of Nominated Members Specific Jump up ^ `` Statewise List '' . 164.100. 47.5 . Retrieved 12 June 2016 . Jump up ^
http://164.100.47.5/Newmembers/partymemberlist.aspx ^ Jump up to : `` Statewise Retirement '' . 164.100. 47.5 . Retrieved 12 June 2016 .
</item 10>

<item 11>
List of nominated members of Rajya Sabha
Sr. No . Image Name Tenure -- Alladi Krishnaswamy Iyer 1952 -- 1953 Satyendranath Bose 1952 -- 1960 Prithviraj Kapoor 1952 -- 1960 -- Jagadisan Mohandas
Kumarappa 1952 -- 1954 5 -- Kalidas Nag 1952 -- 1954 6 Rukmini Devi Arundale 1952 -- 1962 7 -- N.R. Malkani 1952 -- 1962 8 -- Sahib Singh Sokhey 1952 -- 1956 9
-- Zakir Hussain 1952 -- 1962 10 -- Maithili Sharan Gupt 1952 -- 1964 11 -- Kakasaheb Kalelkar 1952 -- 1964 12 -- Radha Kumud Mukherjee 1952 -- 1958 13 P.V.
</item 11>
=== END ITEM id=4a074ad4d325e79c ===
