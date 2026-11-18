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

=== ITEM id=b5d316beb3b1ded4 ===
Gold sentences and facts:
Sentence 0: During the 1995 -- 96 season , Alan Shearer became the first player to score 100 Premier League goals , and holds the record for the fewest games taken to reach 100 , doing so in 124 appearances . He also holds the record for most goals scored in the Premier League . After Shearer , Harry Kane is the second - fastest to 100 goals , doing so in 141 games .
  Fact 0.0: During the 1995-96 season, Alan Shearer became the first player to score 100 Premier League goals.
  Fact 0.1: Alan Shearer holds the record for the fewest games taken to reach 100 goals.
  Fact 0.2: Alan Shearer reached 100 goals in 124 appearances.
  Fact 0.3: Alan Shearer holds the record for most goals scored in the Premier League.
  Fact 0.4: Harry Kane is the second-fastest player to reach 100 goals.
  Fact 0.5: Harry Kane reached 100 goals in 141 games.
Sentence 1: List of Premier League players with 100 or more goals Rank Player Position Premier League club ( s ) Goals Played Ratio Shearer , Alan Alan Shearer Blackburn Rovers , Newcastle United 260 441 0.59 Rooney , Wayne Wayne Rooney Everton , Manchester United 208 484 0.43 Cole , Andy Andy Cole Newcastle United , Manchester United , Blackburn Rovers , Fulham , Manchester City , Portsmouth 187 414 0.45 Lampard , Frank Frank Lampard MF West Ham United , Chelsea , Manchester City 177 609 0.29 Henry , Thierry Thierry Henry Arsenal 175 258 0.68 6 Fowler , Robbie Robbie Fowler Liverpool , Leeds United , Manchester City 163 379 0.43 7 Defoe , Jermain Jermain Defoe West Ham United , Portsmouth , Tottenham Hotspur , Sunderland , Bournemouth 162 485 0.33 8 Owen , Michael Michael Owen Liverpool , Newcastle United , Manchester United , Stoke City 150 326 0.46 9 Ferdinand , Les Les Ferdinand Queens Park Rangers , Newcastle United , Tottenham Hotspur , West Ham United , Leicester City , Bolton Wanderers 149 351 0.42 10 Sheringham , Teddy Teddy Sheringham Nottingham Forest , Tottenham Hotspur , Manchester United , Portsmouth , West Ham United 146 418 0.35 11 van Persie , Robin Robin van Persie Arsenal , Manchester United 144 280 0.51 12 Agüero , Sergio Sergio Agüero Manchester City 143 204 0.70 13 Hasselbaink , Jimmy Floyd Jimmy Floyd Hasselbaink Leeds United , Chelsea , Middlesbrough , Charlton Athletic 127 288 0.44 14 Keane , Robbie Robbie Keane Coventry City , Leeds United , Tottenham Hotspur , Liverpool , West Ham United , Aston Villa 126 349 0.36 15 Anelka , Nicolas Nicolas Anelka Arsenal , Liverpool , Manchester City , Bolton Wanderers , Chelsea , West Bromwich Albion 125 364 0.34 16 Yorke , Dwight Dwight Yorke Aston Villa , Manchester United , Blackburn Rovers , Birmingham City , Sunderland 123 375 0.33 17 Gerrard , Steven Steven Gerrard MF Liverpool 120 504 0.24 18 Wright , Ian Ian Wright Arsenal , West Ham United 113 213 0.53 19 Dublin , Dion Dion Dublin Manchester United , Coventry City , Aston Villa 111 312 0.36 20 Heskey , Emile Emile Heskey Leicester City , Liverpool , Birmingham City , Wigan Athletic , Aston Villa 110 516 0.21 21 Giggs , Ryan Ryan Giggs MF Manchester United 109 632 0.17 22 Crouch , Peter Peter Crouch Aston Villa , Southampton , Liverpool , Portsmouth , Tottenham Hotspur , Stoke City 107 454 0.23 22 Scholes , Paul Paul Scholes MF Manchester United 107 499 0.21 24 Bent , Darren Darren Bent Ipswich Town , Charlton Athletic , Tottenham Hotspur , Sunderland , Aston Villa , Fulham 106 276 0.38 25 Kane , Harry Harry Kane Tottenham Hotspur , Norwich City 104 149 0.70 26 Drogba , Didier Didier Drogba Chelsea 104 254 0.41 27 Lukaku , Romelu Romelu Lukaku West Bromwich Albion , Everton , Manchester United 101 216 0.47 28 Le Tissier , Matt Matt Le Tissier MF Southampton 100 270 0.37 Italics show players still playing professional football ; Bold shows players still playing in the Premier League .
  Fact 1.0: The list is of Premier League players with 100 or more goals.
  Fact 1.1: The table columns are rank, player, position, Premier League club(s), goals, games played, and ratio.
  Fact 1.2: Alan Shearer played for Blackburn Rovers and Newcastle United, scoring 260 goals in 441 games (ratio 0.59).
  Fact 1.3: Wayne Rooney played for Everton and Manchester United, scoring 208 goals in 484 games (ratio 0.43).
  Fact 1.4: Andy Cole played for Newcastle United, Manchester United, Blackburn Rovers, Fulham, Manchester City and Portsmouth, scoring 187 goals in 414 games (ratio 0.45).
  Fact 1.5: Frank Lampard (position MF) played for West Ham United, Chelsea and Manchester City, scoring 177 goals in 609 games (ratio 0.29).
  Fact 1.6: Thierry Henry played for Arsenal, scoring 175 goals in 258 games (ratio 0.68).
  Fact 1.7: Rank 6: Robbie Fowler played for Liverpool, Leeds United and Manchester City, scoring 163 goals in 379 games (ratio 0.43).
  Fact 1.8: Rank 7: Jermain Defoe played for West Ham United, Portsmouth, Tottenham Hotspur, Sunderland and Bournemouth, scoring 162 goals in 485 games (ratio 0.33).
  Fact 1.9: Rank 8: Michael Owen played for Liverpool, Newcastle United, Manchester United and Stoke City, scoring 150 goals in 326 games (ratio 0.46).
  Fact 1.10: Rank 9: Les Ferdinand played for Queens Park Rangers, Newcastle United, Tottenham Hotspur, West Ham United, Leicester City and Bolton Wanderers, scoring 149 goals in 351 games (ratio 0.42).
  Fact 1.11: Rank 10: Teddy Sheringham played for Nottingham Forest, Tottenham Hotspur, Manchester United, Portsmouth and West Ham United, scoring 146 goals in 418 games (ratio 0.35).
  Fact 1.12: Rank 11: Robin van Persie played for Arsenal and Manchester United, scoring 144 goals in 280 games (ratio 0.51).
  Fact 1.13: Rank 12: Sergio Aguero played for Manchester City, scoring 143 goals in 204 games (ratio 0.70).
  Fact 1.14: Rank 13: Jimmy Floyd Hasselbaink played for Leeds United, Chelsea, Middlesbrough and Charlton Athletic, scoring 127 goals in 288 games (ratio 0.44).
  Fact 1.15: Rank 14: Robbie Keane played for Coventry City, Leeds United, Tottenham Hotspur, Liverpool, West Ham United and Aston Villa, scoring 126 goals in 349 games (ratio 0.36).
  Fact 1.16: Rank 15: Nicolas Anelka played for Arsenal, Liverpool, Manchester City, Bolton Wanderers, Chelsea and West Bromwich Albion, scoring 125 goals in 364 games (ratio 0.34).
  Fact 1.17: Rank 16: Dwight Yorke played for Aston Villa, Manchester United, Blackburn Rovers, Birmingham City and Sunderland, scoring 123 goals in 375 games (ratio 0.33).
  Fact 1.18: Rank 17: Steven Gerrard (position MF) played for Liverpool, scoring 120 goals in 504 games (ratio 0.24).
  Fact 1.19: Rank 18: Ian Wright played for Arsenal and West Ham United, scoring 113 goals in 213 games (ratio 0.53).
  Fact 1.20: Rank 19: Dion Dublin played for Manchester United, Coventry City and Aston Villa, scoring 111 goals in 312 games (ratio 0.36).
  Fact 1.21: Rank 20: Emile Heskey played for Leicester City, Liverpool, Birmingham City, Wigan Athletic and Aston Villa, scoring 110 goals in 516 games (ratio 0.21).
  Fact 1.22: Rank 21: Ryan Giggs (position MF) played for Manchester United, scoring 109 goals in 632 games (ratio 0.17).
  Fact 1.23: Rank 22: Peter Crouch played for Aston Villa, Southampton, Liverpool, Portsmouth, Tottenham Hotspur and Stoke City, scoring 107 goals in 454 games (ratio 0.23).
  Fact 1.24: Rank 22: Paul Scholes (position MF) played for Manchester United, scoring 107 goals in 499 games (ratio 0.21).
  Fact 1.25: Rank 24: Darren Bent played for Ipswich Town, Charlton Athletic, Tottenham Hotspur, Sunderland, Aston Villa and Fulham, scoring 106 goals in 276 games (ratio 0.38).
  Fact 1.26: Rank 25: Harry Kane played for Tottenham Hotspur and Norwich City, scoring 104 goals in 149 games (ratio 0.70).
  Fact 1.27: Rank 26: Didier Drogba played for Chelsea, scoring 104 goals in 254 games (ratio 0.41).
  Fact 1.28: Rank 27: Romelu Lukaku played for West Bromwich Albion, Everton and Manchester United, scoring 101 goals in 216 games (ratio 0.47).
  Fact 1.29: Rank 28: Matt Le Tissier (position MF) played for Southampton, scoring 100 goals in 270 games (ratio 0.37).
  Fact 1.30: Italics mark players still playing professional football.
  Fact 1.31: Bold marks players still playing in the Premier League.

Retrieved items:
<item 1>
List of Premier League players with 100 or more goals
During the 1995 -- 96 season , Alan Shearer became the first player to score 100 Premier League goals , and holds the record for the fewest games taken to reach
100 , doing so in 124 appearances . He also holds the record for most goals scored in the Premier League . After Shearer , Harry Kane is the second - fastest to
100 goals , doing so in 141 games . Players ( edit ) As of 18 April 2018 . Sources : List of Premier League players with 100 or more goals Rank Player Position
Premier League club ( s ) Goals Played Ratio Shearer , Alan Alan Shearer Blackburn Rovers , Newcastle United 260 441 0.59 Rooney , Wayne Wayne Rooney Everton ,
Manchester United 208 484 0.43 Cole , Andy Andy Cole Newcastle United , Manchester United , Blackburn Rovers , Fulham , Manchester City , Portsmouth 187 414
0.45 Lampard , Frank Frank Lampard MF West Ham United , Chelsea , Manchester City 177 609 0.29 Henry , Thierry Thierry Henry Arsenal 175 258 0.68 6 Fowler ,
Robbie Robbie Fowler Liverpool ,
</item 1>

<item 2>
Graph query row:
player: Wayne Rooney
relationship: RELATED_TO
related: Alan Shearer
fact: Wayne Rooney was behind only Alan Shearer in Premier League goals of 200 or more
result: None
outcome: None
context: None
</item 2>

<item 3>
List of Premier League players with 100 or more goals
gulf Associated competitions Asia Trophy FA Community Shield FA Cup EFL Cup UEFA Champions League UEFA Europa League Premier League 2 U18 Premier League Premier
League International Cup Retrieved from `` https://en.wikipedia.org/w/index.php?title=List_of_Premier_League_players_with_100_or_more_goals&oldid=837185393 ''
Categories : Lists of Premier League players Association football records and statistics Hidden categories : Wikipedia pending changes protected pages Use dmy
dates from March 2012 Articles with hCards Talk About Wikipedia Bosanski Dansk Español Italiano Shqip Українська Edit links This page was last edited on 19
April 2018 , at 07 : 24 . About Wikipedia
</item 3>

<item 4>
List of Premier League players with 100 or more goals
68 6 Fowler , Robbie Robbie Fowler Liverpool , Leeds United , Manchester City 163 379 0.43 7 Defoe , Jermain Jermain Defoe West Ham United , Portsmouth ,
Tottenham Hotspur , Sunderland , Bournemouth 162 485 0.33 8 Owen , Michael Michael Owen Liverpool , Newcastle United , Manchester United , Stoke City 150 326
0.46 9 Ferdinand , Les Les Ferdinand Queens Park Rangers , Newcastle United , Tottenham Hotspur , West Ham United , Leicester City , Bolton Wanderers 149 351
0.42 10 Sheringham , Teddy Teddy Sheringham Nottingham Forest , Tottenham Hotspur , Manchester United , Portsmouth , West Ham United 146 418 0.35 11 van Persie
, Robin Robin van Persie Arsenal , Manchester United 144 280 0.51 12 Agüero , Sergio Sergio Agüero Manchester City 143 204 0.70 13 Hasselbaink , Jimmy Floyd
Jimmy Floyd Hasselbaink Leeds United , Chelsea , Middlesbrough , Charlton Athletic 127 288 0.44 14 Keane ,
</item 4>

<item 5>
List of Premier League players with 100 or more goals
Bold shows players still playing in the Premier League . See also ( edit ) English football first tier top scorers List of top Premier League goal scorers by
season List of Premier League players with 500 or more appearances References ( edit ) Jump up ^ `` Bent poised to join 100 club '' . PremierLeague.com .
Premier League . 30 January 2012 . Archived from the original on 20 October 2013 . Retrieved 2 February 2012 . Jump up ^ `` Bent delighted to join 100 goals
club '' . PremierLeague.com . Premier League . 2 February 2012 . Retrieved 2 February 2012 . Jump up ^ Dan Kilpatrick ( 4 February 2018 ) . `` Tottenham striker
Harry Kane scores 100th Premier League goal '' . ESPN . ^ Jump up to : `` Premier League Player Stats : Goals '' . PremierLeague.com . Premier League .
Retrieved 10 December 2017 . Jump up ^ `` Premier League Players -- Overview and Stats '' . PremierLeague.com . Premier League .
</item 5>

<item 6>
List of Premier League players with 100 or more goals
PremierLeague.com . Premier League . Retrieved 10 December 2017 . Premier League Seasons 1992 -- 93 1993 -- 94 1994 -- 95 1995 -- 96 1996 -- 97 1997 -- 98 1998
-- 99 1999 -- 2000 2000 -- 01 2001 -- 02 2002 -- 03 2003 -- 04 2004 -- 05 2005 -- 06 2006 -- 07 2007 -- 08 2008 -- 09 2009 -- 10 2010 -- 11 2011 -- 12 2012 --
13 2013 -- 14 2014 -- 15 2015 -- 16 2016 -- 17 2017 -- 18 2018 -- 19 Clubs 2017 -- 18 AFC Bournemouth Arsenal Brighton & Hove Albion Burnley Chelsea Crystal
Palace Everton Huddersfield Town Leicester City Liverpool Manchester City Manchester United Newcastle United Southampton Stoke City Swansea City Tottenham
Hotspur Watford West Bromwich Albion West Ham
</item 6>

<item 7>
List of Premier League players with 100 or more goals
Ryan Ryan Giggs MF Manchester United 109 632 0.17 22 Crouch , Peter Peter Crouch Aston Villa , Southampton , Liverpool , Portsmouth , Tottenham Hotspur , Stoke
City 107 454 0.23 22 Scholes , Paul Paul Scholes MF Manchester United 107 499 0.21 24 Bent , Darren Darren Bent Ipswich Town , Charlton Athletic , Tottenham
Hotspur , Sunderland , Aston Villa , Fulham 106 276 0.38 25 Kane , Harry Harry Kane Tottenham Hotspur , Norwich City 104 149 0.70 26 Drogba , Didier Didier
Drogba Chelsea 104 254 0.41 27 Lukaku , Romelu Romelu Lukaku West Bromwich Albion , Everton , Manchester United 101 216 0.47 28 Le Tissier , Matt Matt Le
Tissier MF Southampton 100 270 0.37 Italics show players still playing professional football ; Bold shows players still playing in the Premier League .
</item 7>

<item 8>
List of Premier League players with 100 or more goals
Charlton Athletic 127 288 0.44 14 Keane , Robbie Robbie Keane Coventry City , Leeds United , Tottenham Hotspur , Liverpool , West Ham United , Aston Villa 126
349 0.36 15 Anelka , Nicolas Nicolas Anelka Arsenal , Liverpool , Manchester City , Bolton Wanderers , Chelsea , West Bromwich Albion 125 364 0.34 16 Yorke ,
Dwight Dwight Yorke Aston Villa , Manchester United , Blackburn Rovers , Birmingham City , Sunderland 123 375 0.33 17 Gerrard , Steven Steven Gerrard MF
Liverpool 120 504 0.24 18 Wright , Ian Ian Wright Arsenal , West Ham United 113 213 0.53 19 Dublin , Dion Dion Dublin Manchester United , Coventry City , Aston
Villa 111 312 0.36 20 Heskey , Emile Emile Heskey Leicester City , Liverpool , Birmingham City , Wigan Athletic , Aston Villa 110 516 0.21 21 Giggs , Ryan Ryan
Giggs MF Manchester United 109 632 0.17 22 Crouch ,
</item 8>

<item 9>
List of Premier League players with 100 or more goals
List of Premier League players with 100 or more goals - wikipedia List of Premier League players with 100 or more goals This is the latest accepted revision ,
reviewed on 19 April 2018 . Jump to : navigation , search See also : List of footballers in England and Scotland by number of league goals Alan Shearer was the
first player to score 200 Premier League goals , and only player to score 100 goals for two clubs . Wayne Rooney is the second Premier League player to score
200 goals or more , behind only Shearer and the only active player as of 2018 to do so . Frank Lampard is the Premier League 's highest scoring midfielder of
all time . Thierry Henry scored 175 Premier League goals , the most by a foreign player . Ryan Giggs is the only player to have scored in each of the first 21
Premier League seasons . Didier Drogba is the only African player to score 100 Premier League goals . Since the Premier League 's formation at the start of the
1992 -- 93 season , 28 players have managed to accrue 100 or more goals in the competition .
</item 9>

<item 10>
List of Premier League players with 100 or more goals
Newcastle United Southampton Stoke City Swansea City Tottenham Hotspur Watford West Bromwich Albion West Ham United Former Aston Villa Barnsley Birmingham City
Blackburn Rovers Blackpool Bolton Wanderers Bradford City Cardiff City Charlton Athletic Coventry City Derby County Fulham Hull City Ipswich Town Leeds United
Middlesbrough Norwich City Nottingham Forest Oldham Athletic Portsmouth Queens Park Rangers Reading Sheffield United Sheffield Wednesday Sunderland Swindon Town
Wigan Athletic Wimbledon ( defunct ) Wolverhampton Wanderers ( returning in 2018 - 19 ) Competition Teams winners Players foreign foreign scorers winners
Managers Stadia Founding Broadcasting foundation Broadcasters Referees Statistics and awards Records All - time table Hat - tricks Highest scores Golden Boot
Golden Glove Manager of the Season Player of the Season Manager of the Month Player of the Month Goal of the Month Players with 500 + appearances Players with
100 + goals Top scorers by season Goalkeepers with 100 + clean sheets 10 Seasons Awards 20 Seasons Awards Finances Richest clubs : Deloitte list Forbes ' list
Team owners Transfer records Game 39 Premier League -- Football League gulf Associated competitions Asia Trophy FA Community Shield FA Cup EFL Cup UEFA
Champions League UEFA Europa League
</item 10>
=== END ITEM id=b5d316beb3b1ded4 ===

=== ITEM id=fcb37204df29ddc0 ===
Gold sentences and facts:
Sentence 0: The American entry into World War I came in April 1917 , after more than two and a half years of efforts by President Woodrow Wilson to keep the United States out of the war . Apart from an Anglophile element urging early support for the British , American public opinion reflected that of the president : the sentiment for neutrality was particularly strong among Irish Americans , German Americans and Scandinavian Americans , as well as among church leaders and among women in general . On the other hand , even before World War I had broken out , American opinion had been more negative toward Germany than towards any other country in Europe . Over time , especially after reports of atrocities in Belgium in 1914 and following the sinking of the passenger liner RMS Lusitania in 1915 , the American people increasingly came to see Germany as the aggressor in Europe .
  Fact 0.0: The American entry into World War I came in April 1917.
  Fact 0.1: The American entry into World War I came after more than two and a half years of efforts by President Woodrow Wilson to keep the United States out of the war.
  Fact 0.2: Apart from an Anglophile element urging early support for the British, American public opinion reflected that of the president.
  Fact 0.3: The sentiment for neutrality was particularly strong among Irish Americans.
  Fact 0.4: The sentiment for neutrality was particularly strong among German Americans.
  Fact 0.5: The sentiment for neutrality was particularly strong among Scandinavian Americans.
  Fact 0.6: The sentiment for neutrality was particularly strong among church leaders.
  Fact 0.7: The sentiment for neutrality was particularly strong among women in general.
  Fact 0.8: Before World War I had broken out, American opinion had been more negative toward Germany than toward any other country in Europe.
  Fact 0.9: Over time, the American people increasingly came to see Germany as the aggressor in Europe.
  Fact 0.10: The shift toward seeing Germany as the aggressor was especially driven by reports of atrocities in Belgium in 1914.
  Fact 0.11: The shift toward seeing Germany as the aggressor was especially driven by the sinking of the passenger liner RMS Lusitania in 1915.

Retrieved items:
<item 1>
American entry into World War I
International Encyclopedia of the First World War . NY Times main headline , April 2 , 1917 , President Calls for War Declaration , Stronger Navy , New Army of
500,000 Men , Full Cooperation With Germany 's Foes President Wilson 's War Address Map of Europe at the time of the US declaration of war on Germany at
omniatlas.com World War I : Declarations of War from Around the Globe - How America Entered the Great War Today in History : U.S.
</item 1>

<item 2>
American entry into World War I
Germany also made a secret offer to help Mexico regain territories lost in the Mexican -- American War in an encoded telegram known as the Zimmermann Telegram ,
which was intercepted by British Intelligence . Publication of that communique outraged Americans just as German U-boats started sinking American merchant ships
in the North Atlantic . Wilson then asked Congress for `` a war to end all wars '' that would `` make the world safe for democracy '' , and Congress voted to
declare war on Germany on April 6 , 1917 . On December 7 , 1917 , the U.S. declared war on Austria - Hungary . U.S. troops began arriving on the Western Front
in large numbers in 1918 . Contents ( hide ) 1 Main issues 1.1 Naval blockade 1.1. 1 The strategy behind the blockade 1.2 Business considerations 1.3 Views of
the elites 2 Public opinion 2.1 Parties 2.2 Workers , farmers , and African Americans 2.3 South 2.4 German Americans 2.5 Christian churches and pacifists 2.6
Jewish Americans 2.7 Irish - Americans 2.
</item 2>

<item 3>
American entry into World War I
American entry into World war I - wikipedia American entry into World war I Jump to : navigation , search The American entry into World War I came in April 1917
, after more than two and a half years of efforts by President Woodrow Wilson to keep the United States out of the war . Apart from an Anglophile element urging
early support for the British , American public opinion reflected that of the president : the sentiment for neutrality was particularly strong among Irish
Americans , German Americans and Scandinavian Americans , as well as among church leaders and among women in general . On the other hand , even before World War
I had broken out , American opinion had been more negative toward Germany than towards any other country in Europe . Over time , especially after reports of
atrocities in Belgium in 1914 and following the sinking of the passenger liner RMS Lusitania in 1915 , the American people increasingly came to see Germany as
the aggressor in Europe . As U.S.
</item 3>

<item 4>
American entry into World War I
Bourne believes elites knew full well what going to war would entail and the price in American lives it would cost . If American elites could portray the United
States ' role in the war as noble , they could convince the generally isolationist American public war would be acceptable . Above all , American attitudes
towards Germany focused on the U-boats ( submarines ) , which sank the Lusitania in 1915 and other passenger ships `` without warning '' . That appeared to
Americans as an unacceptable challenge to America 's rights as a neutral country , and as an unforgivable affront to humanity . After repeated diplomatic
protests , Germany agreed to stop . But in 1917 the Germany military leadership decided that `` military necessity '' dictated the unrestricted use of their
submarines . The Kaiser 's advisors felt America was enormously powerful economically but too weak militarily to make a difference . Declaration of war ( edit )
Germany ( edit ) On April 2 , 1917 , Wilson asked a special joint session of Congress to declare war on the German Empire , stating , `` We have no selfish ends
to serve '' .
</item 4>

<item 5>
American entry into World War I
Sinking of American merchant ships ( edit ) In early 1917 Berlin forced the issue . Its declared decision on 31 January 1917 to target neutral shipping in a
designated war - zone became the immediate cause of the entry of the United States into the war . Five American merchant ships went down in March . Outraged
public opinion now overwhelmingly supported Wilson when he asked Congress for a declaration of war on April 2 , 1917 . Public opinion , moralism , and national
interest ( edit ) Historians such as Ernest R. May have approached the process of American entry into the war as a study in how public opinion changed radically
in three years ' time . In 1914 most Americans called for neutrality , seeing the war a dreadful mistake and were determined to stay out . By 1917 the same
public felt just as strongly that going to war was both necessary and wise . Military leaders had little to say during this debate , and military considerations
were seldom raised . The decisive questions dealt with morality and visions of the future . The prevailing attitude was that America possessed a superior moral
position as the only great nation devoted to the principles of freedom and democracy .
</item 5>

<item 6>
American entry into World War I
252 -- 282 . ^ Jump up to : H.J. Res. 169 : Declaration of War with Austria - Hungary , WWI , United States Senate ^ Jump up to : Jennifer K. Elsea ; Matthew C.
Weed ( April 18 , 2014 ) . `` Declarations of War and Authorizations for the Use of Military Force : Historical Background and Legal Implications '' ( PDF ) .
p. 9 . Archived from the original ( PDF ) on August 10 , 2006 . Retrieved July 21 , 2014 . Jump up ^ Eric W. Osborne , Britain 's economic blockade of Germany ,
1914 -- 1919 ( 2004 ) Jump up ^ Ernest May , The World War and American Isolation , 1914 -- 1917 ( 1959 ) p. 115 quote from Dec 1914 . Jump up ^ C.R.M.F.
Cruttwell , A history of the Great War , 1914 -- 1918 ( 2007 ) p. 191 Jump up ^ Edward House , The intimate papers of Colonel House : Vol 2 ( 1928 ) p.
</item 6>

<item 7>
American entry into World War I
The solution was a third route , a `` peace without victory '' , according to Wilson . On April 6 , 1917 , Congress declared war . In the Senate , the
resolution passed 82 to 6 , with Senators Harry Lane , William J. Stone , James Vardaman , Asle Gronna , Robert M. La Follette , Sr. , and George W. Norris
voting against it . In the House , the declaration passed 373 to 50 , with Claude Kitchin , a senior Democrat , notably opposing it . Another opponent was
Jeannette Rankin , who alone voted against entry into both World War I and World War II . Nearly all of the opposition came from the West and the Midwest .
Austria - Hungary ( edit ) The United States Senate , in a 74 to 0 vote , declared war on Austria - Hungary on December 7 , 1917 , citing Austria - Hungary 's
severing of diplomatic relations with the United States , its use of unrestricted submarine warfare and its alliance with Germany . The declaration passed in
the United States House of Representatives by a vote of 365 to 1 .
</item 7>

<item 8>
American entry into World War I
On April 22 , the German Imperial Embassy warned U.S. citizens against boarding vessels to Britain , which would have to face German attack . On May 7 , Germany
torpedoed the British passenger liner RMS Lusitania , sinking her . This act of aggression caused the loss of 1,198 civilian lives , including 128 Americans .
The sinking of a large , unarmed passenger ship , combined with the previous stories of atrocities in Belgium , shocked Americans and turned public opinion
hostile to Germany , although not yet to the point of war . Wilson issued a warning to Germany that it would face `` strict accountability '' if it sank more
neutral U.S. passenger ships . Berlin acquiesced , ordering its submarines to avoid passenger ships . By January 1917 , however , Field Marshal Paul von
Hindenburg and General Erich Ludendorff decided that an unrestricted submarine blockade was the only way to achieve a decisive victory . They demanded that
Kaiser Wilhelm order unrestricted submarine warfare be resumed . Germany knew this decision meant war with the United States , but they gambled that they could
win before America 's potential strength could be mobilized .
</item 8>

<item 9>
American entry into World War I
Germany 's U-boat warfare was paramount . Had it not been for Berlin 's announcement of January 31 , the president probably would not have issued a call to arms
. Jump up ^ Justus D. Doenecke , Nothing Less Than War : A New History of America 's Entry into World War I ( 2011 ) ch 10 ^ Jump up to : May , Ernest R. ( 1966
) . The World War and American isolation , 1914 -- 1917 . Jump up ^ Knock , Thomas J. ( 1995 ) . To End All Wars : Woodrow Wilson and the Quest for a New World
Order . Jump up ^ Syrett , Harold C. ( 1945 ) . `` The Business Press and American Neutrality , 1914 -- 1917 '' . Mississippi Valley Historical Review . 32 ( 2
) : 215 -- 230 . doi : 10.2307 / 1898209 . JSTOR 1898209 .
</item 9>

<item 10>
American entry into World War I
After February 1 , the communique noted , German U-boats would sink without warning belligerent and neutral ships found in a designated zone comprising waters
around Great Britain , France , and Italy , and in the eastern Mediterranean . The Admiralty made one minor exception : it would permit one American steamer a
week to sail between New York and Falmouth ( ... ) . Initially Germany would grant a period of grace , during which its submarines would not harm neutral ships
that either were en route to the war zone or had already arrived . Jump up ^ Compare : Doenecke , Justus D. ( 2011 ) . Nothing Less Than War : A New History of
America 's Entry Into World War I. Studies in Conflict , Diplomacy and Peace Series . University Press of Kentucky . p. 286 . ISBN 978 - 0 - 8131 - 3002 - 6 .
Retrieved 2015 - 12 - 07 . Several factors led to Wilson 's choice . Germany 's U-boat warfare was paramount .
</item 10>
=== END ITEM id=fcb37204df29ddc0 ===

=== ITEM id=f9b40b97fe8f71a8 ===
Gold sentences and facts:
Sentence 0: Arthur `` The President '' Flanders -- An inmate on death row , convicted of killing his father in an insurance - fraud scheme . His sentence is commuted to life imprisonment , but he is later murdered by an unknown inmate in the prison laundry . Arlen `` The Chief '' Bitterbuck -- A Washita Cherokee death - row inmate , convicted of killing a man in a drunken brawl over a pair of boots . His execution is the first of three mentioned in Paul 's story . Janice Edgecombe -- Paul Edgecombe 's wife . Dies in a bus accident on the way to her grandchild 's graduation in 1956 . Hal Moores -- The warden at Cold Mountain Penitentiary . Melinda Moores -- Warden Moores 's wife , who is dying of a brain tumor and is cured by John Coffey . She eventually dies of a heart attack ten years later . Curtis Anderson -- The assistant warden . Dean Stanton -- A guard on E Block who is strangled and nearly killed by William Wharton . A father of young children , he takes no part in the taking of John Coffey to Melinda Moores due to the risk of losing his job . He applies for relocation to C Block after John Coffey 's death , where he is murdered by an inmate 4 months later . Harry Terwilliger -- One of the main guards on E Block along with his friends Paul Edgecombe , Brutus Howell and Dean Stanton . He eventually dies of cancer , long after John Coffey 's execution . Bill Dodge -- A `` floater '' guard on E Block ( not permanently assigned there ) . Jack Van Hay -- A guard who is part of the execution team . He operates the switch room . Toot - Toot -- A trustee who stands in for the condemned during execution rehearsals and sells snacks to prisoners and guards . Burt Hammersmith -- A reporter who wrote on the Detterick twins ' murders and John Coffey 's trial . Despite believing himself to be an `` Enlightenment '' man , he displays prejudice in his stance on `` Negroes '' and tries to convince Paul of John Coffey 's guilt . Elaine Connelly -- A friend of Paul in the present - day nursing home where he tells his story . She is later revealed to be the grandmother of the Speaker of the Georgia House of Representatives , using this position to scare Brad Dolan from harassing Paul . Homer Cribus -- The Sheriff of Trapingus County , where the murders of the Detterick twins took place . Despite playing no part in apprehending John Coffey , he later shows up at the execution . An outspoken Baptist with strong racial prejudice and immensely overweight , he later succumbs to a heart attack while having sex with a 17 - year - old African American in his office . Rob McGee -- The Deputy Sheriff of Trapingus County , who led the search party which found John Coffey . While displaying strong doubt about Coffey 's guilt after being shown signs of innocence by Paul Edgecombe , he is still powerless to call for an appeal as he is subordinate to Sheriff Cribus , whom he allegedly hopes to succeed . Brad Dolan -- A malicious nursing home employee who harasses Paul Edgecombe . Paul strongly compares him to Percy Wetmore , and several times mistakes him for Percy , despite the latter dying in 1965 . Kathe and Cora Detterick -- The two young girls whom John Coffey was convicted of raping and murdering . Killed by William Wharton . Klaus and Marjorie Detterick -- The parents of Kathe and Cora .
  Fact 0.0: Arthur 'The President' Flanders is an inmate on death row.
  Fact 0.1: Arthur Flanders was convicted of killing his father in an insurance-fraud scheme.
  Fact 0.2: Arthur Flanders's sentence was commuted to life imprisonment.
  Fact 0.3: Arthur Flanders was later murdered by an unknown inmate.
  Fact 0.4: Arthur Flanders was murdered in the prison laundry.
  Fact 0.5: Arlen 'The Chief' Bitterbuck is a Washita Cherokee death-row inmate.
  Fact 0.6: Arlen Bitterbuck was convicted of killing a man in a drunken brawl over a pair of boots.
  Fact 0.7: Arlen Bitterbuck's execution is the first of three mentioned in Paul's story.
  Fact 0.8: Janice Edgecombe is Paul Edgecombe's wife.
  Fact 0.9: Janice Edgecombe dies in a bus accident on the way to her grandchild's graduation.
  Fact 0.10: Janice Edgecombe's death occurred in 1956.
  Fact 0.11: Hal Moores is the warden at Cold Mountain Penitentiary.
  Fact 0.12: Melinda Moores is Warden Moores's wife.
  Fact 0.13: Melinda Moores is dying of a brain tumor.
  Fact 0.14: John Coffey cures Melinda Moores of the brain tumor.
  Fact 0.15: Melinda Moores dies of a heart attack ten years later.
  Fact 0.16: Curtis Anderson is the assistant warden.
  Fact 0.17: Dean Stanton is a guard on E Block.
  Fact 0.18: William Wharton strangles Dean Stanton and nearly kills him.
  Fact 0.19: Dean Stanton is a father of young children.
  Fact 0.20: Dean Stanton takes no part in taking John Coffey to Melinda Moores due to the risk of losing his job.
  Fact 0.21: Dean Stanton applies for relocation to C Block after John Coffey's death.
  Fact 0.22: Dean Stanton is murdered by an inmate on C Block 4 months later.
  Fact 0.23: Harry Terwilliger is one of the main guards on E Block.
  Fact 0.24: Harry Terwilliger is friends with Paul Edgecombe, Brutus Howell, and Dean Stanton.
  Fact 0.25: Harry Terwilliger dies of cancer.
  Fact 0.26: Harry Terwilliger's death came long after John Coffey's execution.
  Fact 0.27: Bill Dodge is a 'floater' guard on E Block.
  Fact 0.28: Bill Dodge is not permanently assigned to E Block.
  Fact 0.29: Jack Van Hay is a guard who is part of the execution team.
  Fact 0.30: Jack Van Hay operates the switch room.
  Fact 0.31: Toot-Toot is a trustee.
  Fact 0.32: Toot-Toot stands in for the condemned during execution rehearsals.
  Fact 0.33: Toot-Toot sells snacks to prisoners and guards.
  Fact 0.34: Burt Hammersmith is a reporter.
  Fact 0.35: Burt Hammersmith wrote on the Detterick twins' murders.
  Fact 0.36: Burt Hammersmith wrote on John Coffey's trial.
  Fact 0.37: Burt Hammersmith believes himself to be an 'Enlightenment' man.
  Fact 0.38: Burt Hammersmith displays prejudice in his stance on 'Negroes'.
  Fact 0.39: Burt Hammersmith tries to convince Paul Edgecombe of John Coffey's guilt.
  Fact 0.40: Elaine Connelly is a friend of Paul Edgecombe in the present-day nursing home where he tells his story.
  Fact 0.41: Elaine Connelly is later revealed to be the grandmother of the Speaker of the Georgia House of Representatives.
  Fact 0.42: Elaine Connelly uses this position to scare Brad Dolan from harassing Paul.
  Fact 0.43: Homer Cribus is the Sheriff of Trapingus County.
  Fact 0.44: The Detterick twins' murders took place in Trapingus County.
  Fact 0.45: Homer Cribus played no part in apprehending John Coffey.
  Fact 0.46: Homer Cribus later shows up at the execution.
  Fact 0.47: Homer Cribus is an outspoken Baptist.
  Fact 0.48: Homer Cribus has strong racial prejudice.
  Fact 0.49: Homer Cribus is immensely overweight.
  Fact 0.50: Homer Cribus later succumbs to a heart attack while having sex with a 17-year-old African American in his office.
  Fact 0.51: Rob McGee is the Deputy Sheriff of Trapingus County.
  Fact 0.52: Rob McGee led the search party which found John Coffey.
  Fact 0.53: Rob McGee displays strong doubt about Coffey's guilt after being shown signs of innocence by Paul Edgecombe.
  Fact 0.54: Rob McGee is powerless to call for an appeal.
  Fact 0.55: Rob McGee is subordinate to Sheriff Cribus.
  Fact 0.56: Rob McGee allegedly hopes to succeed Sheriff Cribus.
  Fact 0.57: Brad Dolan is a malicious nursing home employee.
  Fact 0.58: Brad Dolan harasses Paul Edgecombe.
  Fact 0.59: Paul Edgecombe strongly compares Brad Dolan to Percy Wetmore.
  Fact 0.60: Paul Edgecombe several times mistakes Brad Dolan for Percy Wetmore.
  Fact 0.61: Percy Wetmore died in 1965.
  Fact 0.62: Kathe Detterick and Cora Detterick are two young girls.
  Fact 0.63: John Coffey was convicted of raping and murdering Kathe and Cora Detterick.
  Fact 0.64: William Wharton killed Kathe and Cora Detterick.
  Fact 0.65: Klaus Detterick and Marjorie Detterick are the parents of Kathe and Cora Detterick.
Sentence 1: Toot - Toot -- A trustee who stands in for the condemned during execution rehearsals and sells snacks to prisoners and guards .
  Fact 1.0: Toot-Toot is a trustee.
  Fact 1.1: Toot-Toot stands in for the condemned during execution rehearsals.
  Fact 1.2: Toot-Toot sells snacks to prisoners and guards.

Retrieved items:
<item 1>
The Green Mile (novel)
This year marks the arrival of John Coffey , a 6 ft 8 in powerfully built black man who has been convicted of raping and murdering two small white girls .
During his time on the Mile , John interacts with fellow prisoners Eduard `` Del '' Delacroix , a Cajun arsonist , rapist , and murderer , and William Wharton (
`` Billy the Kid '' to himself , `` Wild Bill '' to the guards ) , a wild - acting and dangerous multiple murderer who is determined to make as much trouble as
he can before he is executed . Other inhabitants include Arlen Bitterbuck , a Native American convicted of killing a man in a fight over a pair of boots ( also
the first character to die in the electric chair ) ; Arthur Flanders , a real estate executive who killed his father to perpetrate insurance fraud , and whose
sentence is eventually commuted to life imprisonment ( while serving his sentence , he is killed by another inmate in the laundry room ) ; and Mr. Jingles , a
mouse , to whom Del teaches various tricks . Paul and the other guards are irritated throughout the book by Percy Wetmore , a sadistic guard who enjoys
antagonizing the prisoners .
</item 1>

<item 2>
The Green Mile (novel)
He is resurrected by John Coffey after being stomped on by Percy Wetmore . This gives him increased longevity and he finally dies 64 years later . Other
characters ( edit ) Arthur `` The President '' Flanders -- An inmate on death row , convicted of killing his father in an insurance - fraud scheme . His
sentence is commuted to life imprisonment , but he is later murdered by an unknown inmate in the prison laundry . Arlen `` The Chief '' Bitterbuck -- A Washita
Cherokee death - row inmate , convicted of killing a man in a drunken brawl over a pair of boots . His execution is the first of three mentioned in Paul 's
story . Janice Edgecombe -- Paul Edgecombe 's wife . Dies in a bus accident on the way to her grandchild 's graduation in 1956 . Hal Moores -- The warden at
Cold Mountain Penitentiary . Melinda Moores -- Warden Moores 's wife , who is dying of a brain tumor and is cured by John Coffey . She eventually dies of a
heart attack ten years later . Curtis Anderson -- The assistant warden .
</item 2>

<item 3>
The Green Mile (novel)
Despite believing himself to be an `` Enlightenment '' man , he displays prejudice in his stance on `` Negroes '' and tries to convince Paul of John Coffey 's
guilt . Elaine Connelly -- A friend of Paul in the present - day nursing home where he tells his story . She is later revealed to be the grandmother of the
Speaker of the Georgia House of Representatives , using this position to scare Brad Dolan from harassing Paul . Homer Cribus -- The Sheriff of Trapingus County
, where the murders of the Detterick twins took place . Despite playing no part in apprehending John Coffey , he later shows up at the execution . An outspoken
Baptist with strong racial prejudice and immensely overweight , he later succumbs to a heart attack while having sex with a 17 - year - old African American in
his office . Rob McGee -- The Deputy Sheriff of Trapingus County , who led the search party which found John Coffey . While displaying strong doubt about Coffey
's guilt after being shown signs of innocence by Paul Edgecombe , he is still powerless to call for an appeal as he is subordinate to Sheriff Cribus , whom he
allegedly hopes to succeed .
</item 3>

<item 4>
The Green Mile (novel)
Even at the end , during his execution , he asks Paul Edgecombe not to put on the traditional black silk mask used to block the view of the prisoner 's face
because he fears the dark . Coffey is described as `` knowing his own name and not much else '' and lacks the capability to do so much as tie a simple knot .
However , he is convicted of luring the girls away from their home , disposing of the watchdog , carefully planning and using abilities he would otherwise not
be expected to have . He is the calmest and mildest prisoner the guards have ever seen , despite his hulking form . He turns out to be innocent of what he is
accused of , but chooses to die anyway . Percy Wetmore -- He is a young and sadistic guard . He is disliked by Paul and the other guards because of his ways ,
but they can not do anything about it because he is the nephew of the governor 's wife . He is very homophobic and attacks Eduard Delacroix for allegedly
touching him , although it was an accident caused by Del stumbling out of the prison truck .
</item 4>

<item 5>
The Green Mile (novel)
He is the first character to discover John Coffey 's amazing abilities , when the prisoner cures his urinary tract infection . It is also his idea to take
Coffey to try to cure Melinda , Warden Hal Moores ' wife , of her brain tumor . He transferred to Boy 's Correctional with Brutus Howell shortly after Coffey 's
execution . Brutus `` Brutal '' Howell -- He is second in command on the Green Mile and Paul 's best friend . He is a tall , imposing man but not violent at all
unless necessary . His nickname of `` Brutal '' is intended as irony . A former football tackle who had gone on to play at Louisiana State University ( LSU ) in
his youth , he eventually suffers a fatal heart attack at home , in his fifties , about twenty - five years after Coffey 's execution . John Coffey -- He is a
massive black man ( 6 ft 8 in tall ) , on death row for the alleged rape and murder of two young girls . He is very quiet and prefers to keep to himself , weeps
almost constantly , and is afraid of the dark .
</item 5>

<item 6>
The Green Mile (novel)
Paul is unsure how to help John , but John tells him not to worry , as he is ready to die anyway , wanting to escape the cruelty of the world . John 's
execution is the last one in which Paul participates . He introduces Mr. Jingles to Elaine just before the mouse dies , having lived 64 years past these events
, and explains that those healed by John gained an unnaturally long lifespan . Elaine dies shortly after , never learning how Paul 's wife died in his arms
immediately after they suffered a bus accident , and that he then saw John Coffey 's ghost watching him from an overpass . Paul seems to be all alone , now 104
years old , and wondering how much longer he will live . Characters ( edit ) Paul Edgecombe -- The protagonist and narrator of the book and the death - row
supervisor at Cold Mountain Penitentiary . He is 40 years old when the main bulk of the story takes place , in 1932 . He is a caring man and takes excellent
care of the men on his block , avoiding conflict and keeping the peace whenever possible .
</item 6>

<item 7>
The Green Mile (novel)
Curtis Anderson -- The assistant warden . Dean Stanton -- A guard on E Block who is strangled and nearly killed by William Wharton . A father of young children
, he takes no part in the taking of John Coffey to Melinda Moores due to the risk of losing his job . He applies for relocation to C Block after John Coffey 's
death , where he is murdered by an inmate 4 months later . Harry Terwilliger -- One of the main guards on E Block along with his friends Paul Edgecombe , Brutus
Howell and Dean Stanton . He eventually dies of cancer , long after John Coffey 's execution . Bill Dodge -- A `` floater '' guard on E Block ( not permanently
assigned there ) . Jack Van Hay -- A guard who is part of the execution team . He operates the switch room . Toot - Toot -- A trustee who stands in for the
condemned during execution rehearsals and sells snacks to prisoners and guards . Burt Hammersmith -- A reporter who wrote on the Detterick twins ' murders and
John Coffey 's trial .
</item 7>

<item 8>
The Green Mile (novel)
He is later attacked by `` Wild Bill '' Wharton , consequently wets himself , and is teased by Delacroix for it . In retaliation , Percy deliberately sabotages
Delacroix 's execution . At the end of the story he is sent to the Briar Ridge mental institution , originally considered for a job but now as a patient , after
Coffey transferred Melinda 's disease to him which caused him to kill William Wharton . He eventually lives through a hospital fire and dies in 1965 . William
`` Wild Bill '' Wharton -- The antagonist of the story . He is on death row for various crimes . He does not like the nickname `` Wild Bill '' but prefers to be
called `` Billy the Kid '' , a name which he has tattooed on his forearm . When he first arrives he manages to convince the guards that he is in a drugged
stupor , only to attack and attempt to strangle to death Dean Stanton when they reach E Block . He continues to wreak havoc on the Mile and plays tricks such as
urinating on the guards , amongst other things . He is punished by being placed in solitary confinement , but never seems to learn his lesson .
</item 8>

<item 9>
The Green Mile (novel)
He is punished by being placed in solitary confinement , but never seems to learn his lesson . As John Coffey is being smuggled to Hal Moores 's house , Wild
Bill grabs his arm and Coffey sees that he actually committed the murders Coffey was accused of . Therefore , Coffey gives Percy the `` sickness '' he took from
the warden 's wife , causing him to kill Wharton . Eduard `` Del '' Delacroix -- He is a Cajun prisoner with a fairly slow grasp of the English language . He is
incarcerated for multiple deaths due to a fire he started while trying to cover up his rape and murder of a young girl . While on the Mile , Del befriends a
mouse named Mr. Jingles , who becomes his best friend in his last days on death row . Percy , his enemy , sabotages his execution , causing Del to die in a slow
, gruesome death in the electric chair . Mr. Jingles ( nicknamed `` Steamboat Willie '' ) -- An unusually intelligent mouse who enjoys eating peppermint sweets
. He becomes a friend to Eduard Delacroix in the few days before the man is executed .
</item 9>

<item 10>
The Green Mile (novel)
Brad Dolan -- A malicious nursing home employee who harasses Paul Edgecombe . Paul strongly compares him to Percy Wetmore , and several times mistakes him for
Percy , despite the latter dying in 1965 . Kathe and Cora Detterick -- The two young girls whom John Coffey was convicted of raping and murdering . Killed by
William Wharton . Klaus and Marjorie Detterick -- The parents of Kathe and Cora . Reception ( edit ) The Green Mile won the Bram Stoker Award for Best Novel in
1996 . In 1997 , The Green Mile was nominated as Best Novel for the British Fantasy Award and the Locus Award . In 2003 the book was listed on the BBC 's The
Big Read poll of the UK 's `` best - loved novel . '' Film adaptation ( edit ) Main article : The Green Mile ( film ) Frank Darabont adapted the novel into a
screenplay for a feature film of the same name . Released in 1999 , the film was directed by Darabont and starred Tom Hanks as Paul Edgecombe and Michael Clarke
Duncan as John Coffey .
</item 10>
=== END ITEM id=f9b40b97fe8f71a8 ===

=== ITEM id=6a1cc0c31cac6227 ===
Gold sentences and facts:
Sentence 0: The Homestead Acts were several United States federal laws that gave an applicant ownership of land , typically called a `` homestead , '' at no cost . In all , more than 270 million acres of public land , or nearly 10 % of the total area of the U.S. , was given away free to 1.6 million homesteaders ; most of the homesteads were west of the Mississippi River .
  Fact 0.0: The Homestead Acts were several United States federal laws.
  Fact 0.1: The Homestead Acts gave an applicant ownership of land.
  Fact 0.2: The land given under the Homestead Acts is typically called a homestead.
  Fact 0.3: The land was given at no cost.
  Fact 0.4: More than 270 million acres of public land were given away free under the Homestead Acts.
  Fact 0.5: The 270 million acres of public land were nearly 10% of the total area of the U.S.
  Fact 0.6: The public land was given away free to 1.6 million homesteaders.
  Fact 0.7: Most of the homesteads were west of the Mississippi River.
Sentence 1: An extension of the Homestead Principle in law , the Homestead Acts were an expression of the `` Free Soil '' policy of Northerners who wanted individual farmers to own and operate their own farms , as opposed to Southern slave - owners who wanted to buy up large tracts of land and use slave labor , thereby shutting out free white men .
  Fact 1.0: The Homestead Acts were an extension of the Homestead Principle in law.
  Fact 1.1: The Homestead Acts were an expression of the 'Free Soil' policy of Northerners.
  Fact 1.2: Northerners with the Free Soil policy wanted individual farmers to own and operate their own farms.
  Fact 1.3: Southern slave-owners, as opposed to Northerners, wanted to buy up large tracts of land.
  Fact 1.4: Southern slave-owners wanted to use slave labor on large tracts of land.
  Fact 1.5: Southern slave-owners' land purchases would shut out free white men.

Retrieved items:
<item 1>
Homestead Acts
Any adult who had never taken up arms against the U.S. government could apply . Women and immigrants who had applied for citizenship were eligible . The 1866
Act explicitly included black Americans and encouraged them to participate , but rampant discrimination slowed black gains . Historian Michael Lanza argues that
while the 1866 law pack was not as beneficial as it might have been , it was part of the reason that by 1900 one fourth of all Southern black farmers owned
their own farms . Several additional laws were enacted in the latter half of the 19th and early 20th centuries . The Southern Homestead Act of 1866 sought to
address land ownership inequalities in the south during Reconstruction . The Timber Culture Act of 1873 granted land to a claimant who was required to plant
trees -- the tract could be added to an existing homestead claim and had no residency requirement . The Kinkaid Amendment of 1904 granted a full section ( 640
acres ) to new homesteaders settling in western Nebraska .
</item 1>

<item 2>
Homestead Acts
History ( edit ) Donation land claim Act of 1850 ( edit ) Main article : Donation Land Claim Act The Donation Land Claim Act allowed settlers to claim land in
the Oregon Territory , then including the modern states of Washington , Oregon , Idaho and parts of Wyoming . Settlers were able to claim 320 or 640 acres of
land for free between 1850 and 1854 , and then at a cost of $1.25 per acres until the law expired in 1855 . Homestead Act of 1862 ( edit ) Certificate of
homestead in Nebraska given under the Homestead Act , 1862 Wikisource has original text related to this article : Homestead Act ( 1862 ) The `` yeoman farmer ''
ideal of Jeffersonian democracy was still a powerful influence in American politics during the 1840 -- 1850s , with many politicians believing a homestead act
would help increase the number of `` virtuous yeomen '' .
</item 2>

<item 3>
Westward Expansion Trails
Westward expansion trails - wikipedia Westward expansion trails Jump to : navigation , search This article needs additional citations for verification . Please
help improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( March 2013 ) ( Learn how and when to
remove this template message ) In the American Old West , overland trails were popular means of travel used by pioneers and immigrants throughout the 19th
century and especially between 1830 and 1870 as an alternative to sea and railroad transport . These immigrants began to settle various regions of North America
west of the Great Plains as part of the mass overland migrations of the mid-19th century . Settlers emigrating from the eastern United States were spurred by
various motives , among them religious persecution , economic incentives , and the prospect of new wealth and new freedoms in what was then a largely unspoiled
and unregulated country . By the 1840s , several popular and well - established routes had been blazed through the continental interior to destinations in the
far west , including the Oregon Trail , California Trail , and Mormon Trail .
</item 3>

<item 4>
Westward Expansion Trails
During the twenty - five years 1841 -- 1866 , 250,000 to 650,000 people `` pulled up stakes , '' and headed west along these trails . About one - third
immigrated to Oregon , one - third to California and one - third to Utah , Colorado , and Montana . Although it is often stated that the Northern trails began
in certain cities on the Missouri River , emigrants following any of the three trails typically left from one of three `` jumping off '' points on the Missouri
's steamboat serviced river ports : Independence , Missouri or Saint Joseph , Missouri , or Council Bluffs , Iowa . ( Once known as Kanesville , Iowa until 1852
; after river dredging in the early 1850s , the latter town at the Missouri - Platte confluence became the most common departure point since it was close in
proximity to the River Platte -- along which the eastern trails ascend to South Pass above Fort Laramie. ) The trails from these cities ( and several others )
converged in the mostly empty flatlands of central Nebraska near present - day Kearney , in the vicinity of Fort Kearney .
</item 4>

<item 5>
Westward Expansion Trails
After the end of the Mexican -- American War in 1849 , vast new American conquests again enticed mass immigration . Legislation like the Donation Land Claim Act
and significant events like the California Gold Rush further lured colonists to travel overland to the west . Two major wagon - based transportation networks ,
one typically starting in Missouri and the other in the Mexican province of Santa Fe de Nuevo México , served the majority of migrants during the era of
westward expansion . Three of the Missouri - based routes -- the Oregon , Mormon , and California Trails -- were collectively known as the Emigrant Trails .
Historians have estimated at least 500,000 emigrants used these three trails between 1843 and 1869 , and despite growing competition from transcontinental
railroads , some use even continued into the early 20th century . The major southern routes were the Santa Fe Trail , the Southern Emigrant Trail , and the Old
Spanish Trail , as well as its wagon road successor the Mormon Road , a southern spur of the California Trail used in the winter that also made use of the
western half of the Old Spanish Trail .
</item 5>

<item 6>
Westward Expansion Trails
It was the only practical way for settlers in wagons without tools , livestock , or supplies to cross the mountains and usually thought critical to the
colonization of the American West . Some of the first to travel the Oregon Trail were Christian missionaries , members of the Methodist Episcopal Church who
established the Methodist Mission in 1834 . Even though they did n't make many converts , they were impressed by the short amount of time needed to reach the
Pacific Coast . Rumors about how the sun always shone there and wheat grew as tall as a man attracted American settlers . American settlers began following the
trail in 1841 , with the first recorded colonist wagon traingroup being the 1843 `` Great Migration '' of about 900 colonists , led by Marcus Whitman . The
Provisional Government of Oregon was established by such colonists in 1843 , generally limited to the Willamette Valley . The Oregon Treaty of 1846 divided the
Oregon Country between Great Britain and the United States , mostly along the 49th parallel . The Territory of Oregon was established shortly afterwards , in
1848 , and over 12,000 American settlers made the journey there during the decade .
</item 6>

<item 7>
Homestead Acts
The Free Soil Party of 1848 -- 52 , and the new Republican Party after 1854 , demanded that the new lands opening up in the west be made available to
independent farmers , rather than wealthy planters who would develop it with the use of slaves forcing the yeomen farmers onto marginal lands . Southern
Democrats had continually fought ( and defeated ) previous homestead law proposals , as they feared free land would attract European immigrants and poor
Southern whites to the west . After the South seceded and their delegates left Congress in 1861 , the Republicans and other supporters from the upper South
passed a homestead act . The intent of the first Homestead Act , passed in 1862 , was to liberalize the homesteading requirements of the Preemption Act of 1841
. Its leading advocates were Andrew Johnson , George Henry Evans and Horace Greeley . The homestead was an area of public land in the West ( usually 160 acres
or 65 ha ) granted to any US citizen willing to settle on and farm the land for at least five years .
</item 7>

<item 8>
Westward Expansion Trails
From their labors arose the State of Deseret , later to become the Utah Territory , and finally the State of Utah . The Mormon pioneers shared similar
experiences with others traveling west : the drudgery of walking hundreds of miles , suffocating dust , violent thunderstorms , mud , temperature extremes , bad
water , poor forage , sickness , and death . They recorded their experiences in journals , diaries , and letters . The Mormons , however , were a unique part of
this migration . Their move to the Valley of the Great Salt Lake was a response to their violent expulsion from Missouri and Illinois . As it was also motivated
by a desire to maintain a religious and cultural identity it was necessary to find an isolated area where they could permanently settle and practice their
religion in peace . Southern Emigrant Trail ( edit ) Main article : Southern Emigrant Trail The Southern Emigrant Trail was a major land route for immigration
into California from the eastern United States that followed the Santa Fe Trail to New Mexico during the California Gold Rush .
</item 8>

<item 9>
Homestead Acts
The federal government charged royalties for extraction of these resources from public lands . On the other hand , homesteading schemes were generally pointless
for land containing `` locatable minerals , '' such as gold and silver , which could be controlled through mining claims under the Mining Act of 1872 , for
which the federal government did not charge royalties . The government developed no systematic method to evaluate claims under the homestead acts . Land offices
relied on affidavits from witnesses that the claimant had lived on the land for the required period of time and made the required improvements . In practice ,
some of these witnesses were bribed or otherwise colluded with the claimant . Although not necessarily fraud , it was common practice for the eligible children
of a large family to claim nearby land as soon as possible . After a few generations , a family could build up a sizable estate . The homesteads were criticized
as too small for the environmental conditions on the Great Plains ; a homesteader using 19th - century animal - powered tilling and harvesting could not have
cultivated the 1500 acres later recommended for dry land farming .
</item 9>

<item 10>
Homestead Acts
Some scholars believe the acreage limits were reasonable when the act was written , but reveal that no one understood the physical conditions of the plains .
According to Hugh Nibley , much of the rain forest west of Portland , Oregon was acquired by the Oregon Lumber Company by illegal claims under the Act .
Nonetheless , in 1995 , a random survey of 178 members of the Economic History Association found that 70 percent of economists and 84 percent of economic
historians disagreed that `` Nineteenth - century U.S. land policy , which attempted to give away free land , probably represented a net drain on the productive
capacity of the country . '' Related Acts in other countries ( edit ) Canada ( edit ) Similar laws were passed in Canada : The Legislative Assembly of Ontario
passed The Free Grants and Homestead Act in 1868 , which introduced a conditional scheme to an existing free grant plan previously authorized by the Province of
Canada in The Public Lands Act of 1860 .
</item 10>
=== END ITEM id=6a1cc0c31cac6227 ===

=== ITEM id=626d54719018fc37 ===
Gold sentences and facts:
Sentence 0: `` The Glory of Love '' is a song written by Billy Hill , recorded by Benny Goodman in 1936 , whose version was a number one pop hit . In 1951 , R&B vocal group , The Five Keys , had their biggest R&B hit with their version of the song , hitting number one on the R&B chart for four non-consecutive weeks . Although The Five Keys recording sold a reported million copies pressed recordings are very rare .
  Fact 0.0: The Glory of Love is a song.
  Fact 0.1: The Glory of Love was written by Billy Hill.
  Fact 0.2: The Glory of Love was recorded by Benny Goodman in 1936.
  Fact 0.3: Benny Goodman's version of The Glory of Love was a number one pop hit.
  Fact 0.4: In 1951, the R&B vocal group The Five Keys had their biggest R&B hit with their version of The Glory of Love.
  Fact 0.5: The Five Keys' version hit number one on the R&B chart for four non-consecutive weeks.
  Fact 0.6: The Five Keys' recording reportedly sold a million copies.
  Fact 0.7: Pressed recordings of The Five Keys' version are very rare.
Sentence 1: `` The Glory of Love '' Single by Benny Goodman Released 1936 Format 78 rpm vinyl Recorded 1936 Genre Pop Songwriter ( s ) Billy Hill Benny Goodman singles chronology `` The Glory of Love ' '' ( 1936 ) The Glory of Love'1936 The Five Keys singles chronology `` The Glory of Love ' '' ( 1951 ) The Glory of Love'1951 `` Yes Sir , That 's My Baby '' ( 1952 ) Yes Sir , That 's My Baby1952
  Fact 1.0: The Glory of Love is a single by Benny Goodman.
  Fact 1.1: The Glory of Love was released in 1936.
  Fact 1.2: The Glory of Love has the format 78 rpm vinyl.
  Fact 1.3: The Glory of Love was recorded in 1936.
  Fact 1.4: The Glory of Love has the genre pop.
  Fact 1.5: The songwriter of The Glory of Love is Billy Hill.
  Fact 1.6: The Benny Goodman singles chronology lists The Glory of Love (1936).
  Fact 1.7: The Five Keys singles chronology lists The Glory of Love (1951).
  Fact 1.8: The Five Keys singles chronology lists Yes Sir, That's My Baby (1952).

Retrieved items:
<item 1>
Joy to the World (Three Dog Night song)
Sugar '' -- The Archies 1970 : `` Bridge over Troubled Water '' -- Simon & Garfunkel 1971 : `` Joy to the World '' -- Three Dog Night 1972 : `` The First Time
Ever I Saw Your Face '' -- Roberta Flack 1973 : `` Tie a Yellow Ribbon Round the Ole Oak Tree '' -- Tony Orlando and Dawn 1974 : `` The Way We Were '' -- Barbra
Streisand 1975 : `` Love Will Keep Us Together '' -- Captain & Tennille 1976 : `` Silly Love Songs '' -- Wings 1977 : `` Tonight 's the Night ( Gonna Be Alright
) '' -- Rod Stewart 1978 : `` Shadow Dancing '' -- Andy Gibb 1979 : `` My Sharona '' -- The Knack Complete list ( 1946 -- 1959 ) ( 1960 -- 1979 ) ( 1980 -- 1999
) ( 2000 -- 2019 ) Retrieved from `` https://en.wikipedia.org/w/index.
</item 1>

<item 2>
The Glory of Love (song)
Although The Five Keys recording sold a reported million copies pressed recordings are very rare . Contents ( hide ) 1 Charts 1.1 Weekly charts 2 Otis Redding
version 3 Charts 3.1 Weekly charts 4 References Charts ( edit ) Weekly charts ( edit ) Chart ( 1951 ) Peak position US Billboard Hot Rhythm & Blues Songs Otis
Redding version ( edit ) `` The Glory of Love '' Single by Otis Redding from the album The Dock of the Bay B - side `` I 'm Coming Home '' Released 1967 Format
7 '' vinyl Recorded December , 1967 Genre Soul , Pop Length 2 : 38 Label Volt S419 Songwriter ( s ) Billy Hill Producer ( s ) Steve Cropper Otis Redding singles
chronology `` Shake '' ( 1967 ) `` The Glory of Love '' ( 1967 ) `` Tramp '' ( 1967 ) `` Shake '' ( 1967 ) `` The Glory of Love '' ( 1967 ) `` Tramp '' ( 1967 )
In 1967 , Otis Redding recorded a cover version for his 1968 album , The Dock of the Bay .
</item 2>

<item 3>
The Glory of Love (song)
In 1951 , R&B vocal group , The Five Keys , had their biggest R&B hit with their version of the song , hitting number one on the R&B chart for four
non-consecutive weeks . Although The Five Keys recording sold a reported million copies pressed recordings are very rare .
</item 3>

<item 4>
Let My Love Open the Door
Billboard Hot 100 59 U.S. Cashbox 92 Personnel ( edit ) Pete Townshend : vocals , guitars , synths John `` Rabbit '' Bundrick : `` straight '' keyboards Simon
Phillips : drums Tony Butler : bass guitar Ted Jensen - mastering engineer Covers ( edit ) A version of this song by Luminate appears on The Covers EP . It was
also covered by M. Ward ( for the 2005 Starbucks compilation album , Sweetheart 2005 : Love Songs ) Christian rock group Audio Adrenaline covered this song on
their 1999 album Underdog . Natalie Imbruglia covered the song on her album Male released in 2015 . Great Big Sea covered the song on their 2012 album XX .
Roger Daltrey covered the song in 2015 for a campaign involving Teen Cancer America and First Citizens Bank . A full version was released in February 2016 ,
with all proceeds going to Teen Cancer America . The group Rogue Wave also covered it in `` Cover me '' released in 2017 .
</item 4>

<item 5>
Let My Love Open the Door
at 22 : 30 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the
Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 5>

<item 6>
The Glory of Love (song)
Redding 's cover became a top 20 hit and reached number nineteen on the Billboard R&B Songs chart and number sixty on the Billboard Hot 100 . Charts ( edit )
Weekly charts ( edit ) Chart ( 1967 ) Peak position US Billboard Hot 100 60 US Billboard Hot Rhythm & Blues Songs 19 References ( edit ) Jump up ^ Whitburn ,
Joel ( 2004 ) . Top R&B / Hip - Hop Singles : 1942 - 2004 . Record Research . p. 205 . Jump up ^ Propes , Steve ( 1973 ) . Those Oldies But Goodies : A Guide to
50 's Record Collecting . The Macmillan Company , New York . p. 42 . Jump up ^ Whitburn , Joel ( 2004 ) . Top R&B / Hip - Hop Singles : 1942 - 2004 . Record
Research . p. 486 . Jump up ^ `` allmusic ( ( ( Otis Redding > Awards ) ) ) '' . Billboard . Retrieved 2012 - 11 - 22 .
</item 6>

<item 7>
The Glory of Love (song)
The Glory of Love ( song ) - wikipedia The Glory of Love ( song ) Jump to : navigation , search This article needs additional citations for verification .
Please help improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( December 2015 ) ( Learn how and
when to remove this template message ) For the song by Peter Cetera , see Glory of Love . `` The Glory of Love '' Single by Benny Goodman Released 1936 Format
78 rpm vinyl Recorded 1936 Genre Pop Songwriter ( s ) Billy Hill Benny Goodman singles chronology `` The Glory of Love ' '' ( 1936 ) The Glory of Love'1936 The
Five Keys singles chronology `` The Glory of Love ' '' ( 1951 ) The Glory of Love'1951 `` Yes Sir , That 's My Baby '' ( 1952 ) Yes Sir , That 's My Baby1952 ``
The Glory of Love '' is a song written by Billy Hill , recorded by Benny Goodman in 1936 , whose version was a number one pop hit .
</item 7>

<item 8>
The Day the Music Died
ISBN 978 - 0 - 89672 - 365 - 8 . Crouse , Richard ( 2012 ) . Who Wrote The Book of Love ? . Random House Digital . ISBN 978 - 0 - 385 - 67442 - 3 . Everitt ,
Rich ( 2004 ) . Falling Stars : Air Crashes That Filled Rock and Roll Heaven . Harbor House . ISBN 978 - 1 - 891799 - 04 - 4 . McDonald , Les ( 2010 ) . The Day
the Music Died . Xlibris Corporation . ISBN 978 - 1 - 4691 - 1356 - 2 . Jennings , Waylon ; Kaye , Lenny ( 1996 ) . Waylon : An Autobiography . Warner Books .
ISBN 978 - 0 - 446 - 51865 - 9 . Schuck , Raymond ( 2012 ) . Do You Believe in Rock and Roll ? Essays on Don Mclean 's American Pie . McFarland .
</item 8>

<item 9>
The Glory of Love (song)
Billboard . Retrieved 2012 - 11 - 22 . Jump up ^ `` Billboard R&B Singles Chart '' . Billboard . Retrieved 2016 - 01 - 27 . Preceded by `` Sixty Minute Man ''
by The Dominoes Billboard Best Selling Retail Rhythm & Blues Records number - one single ( The Five Keys version ) September 22 , 1951 Succeeded by `` Sixty
Minute Man '' by The Dominoes Retrieved from `` https://en.wikipedia.org/w/index.php?title=The_Glory_of_Love_(song)&oldid=814898703 '' Categories : 1936 songs
1951 singles Benny Goodman songs Otis Redding songs Songs written by Billy Hill ( songwriter ) Billboard Hot R&B / Hip - Hop Songs number - one singles Paul
McCartney songs Hidden categories : Articles needing additional references from December 2015 All articles needing additional references Music infoboxes with
deprecated parameters Articles with hAudio microformats Talk Contents About Wikipedia Norsk Edit links This page was last edited on 11 December 2017 , at 15 :
51 . About Wikipedia
</item 9>

<item 10>
You'll Never Find Another Love Like Mine
Record Research . p. 213 . Jump up ^ `` Billboard Hot 100 '' . 1976 - 09 - 04 . Retrieved 2015 - 06 - 07 . Jump up ^ `` Billboard Hot 100 '' . 1976 - 09 - 11 .
Retrieved 2015 - 06 - 07 . Jump up ^ `` The Hits Just Keep On Comin ' : You 'll Never Find Another Love Like Mine '' . Hitsjustkeeponcomin.blogspot.com. 2005 -
01 - 12 . Retrieved 2016 - 10 - 12 . Jump up ^ ( 1 ) Jump up ^ `` Archived copy '' . Archived from the original on June 2 , 2016 . Retrieved October 27 , 2015 .
Jump up ^ `` Image : RPM Weekly - Library and Archives Canada '' . Bac-lac.gc.ca . Retrieved 2016 - 10 - 12 . Jump up ^ Whitburn , Joel ( 2002 ) . Top Adult
Contemporary : 1961 - 2001 . Record Research .
</item 10>
=== END ITEM id=626d54719018fc37 ===

=== ITEM id=84ec9ee2323f26b2 ===
Gold sentences and facts:
Sentence 0: Season Episodes Originally aired First aired Last aired 17 September 13 , 1969 ( 1969 - 09 - 13 ) January 17 , 1970 ( 1970 - 01 - 17 ) 8 September 12 , 1970 ( 1970 - 09 - 12 ) October 31 , 1970 ( 1970 - 10 - 31 ) 16 September 9 , 1978 December 23 , 1978
  Fact 0.0: The table lists 17 episodes with first airing on September 13, 1969 and last airing on January 17, 1970.
  Fact 0.1: The table lists 8 episodes with first airing on September 12, 1970 and last airing on October 31, 1970.
  Fact 0.2: The table lists 16 episodes with first airing on September 9, 1978 and last airing on December 23, 1978.

Retrieved items:
<item 1>
List of Scooby-Doo, Where Are You! episodes
5 5 `` Decoy for a Dognapper '' Indian Witch Doctor and Ghost of Geronimo / Buck Masters October 11 , 1969 ( 1969 - 10 - 11 ) When a series of local dognappings
alarms prize - winning dog trainer Buck Masters , the gang offers to have Scooby - Doo pose as a decoy to catch the dognappers . However , their plan works too
well as Scooby is himself dognapped , leading the gang to discover that the dognappers appear to be under the command of a ghostly Indian witch doctor . 6 6 ``
What the Hex Going On ? '' Ghost of Elias Kingston / Stuart Weatherby October 18 , 1969 ( 1969 - 10 - 18 ) The kids pay a visit to their friend Sharon Weatherby
, whose Uncle Stuart has been cursed by the ghost of the family 's neighbor , Elias Kingston . When Uncle Stuart disappears altogether , Mystery , Inc. ventures
into the abandoned and allegedly haunted Kingston Mansion next door , in search of him and the ghost .
</item 1>

<item 2>
List of Scooby-Doo, Where Are You! episodes
24 7 `` Who 's Afraid of the Big Bad Werewolf ? '' Ghost of Silas Long the Werewolf / Sheep Rustler Tell Me , Tell Me October 24 , 1970 ( 1970 - 10 - 24 ) Out
on a camping trip , the gang hears the sounds of a wolf , so they decide to follow wolf - like tracks to the open and empty grave of Silas Long , who was
supposedly a werewolf . They follow his tracks further to an old abandoned sawmill filled with wool , empty barrels , and the werewolf ghost himself . 25 8 ``
Do n't Fool with a Phantom '' Wax Phantom / Roger Steven Pretty Mary Sunlight October 31 , 1970 ( 1970 - 10 - 31 ) While participating in the Johnny Sands Dance
Game Show at the local TV station , the teens encounter a phantom made of wax , who steals a safe full of money and kidnaps the station manager . Is the phantom
a hoax , or the result of a curse from an old station employee seeking revenge for the cancellation of his show ?
</item 2>

<item 3>
List of Scooby-Doo, Where Are You! episodes
9 9 `` The Backstage Rage '' Phantom Puppeteer / Mr. Pietro November 8 , 1969 ( 1969 - 11 - 08 ) While getting a pizza , Shaggy and Scooby - Doo come across a
violin case filled with counterfeit money . While Scooby is distracted by a dog puppet , the case of money is stolen away . Shaggy , Scooby , and the rest of
the gang find a string puppet controller at the scene and follow it to the local puppet theater , where they uncover a counterfeiting operation , controlled by
a spooky puppet master . 10 10 `` Bedlam in the Big Top '' Ghost Clown / Harry the Hypnotist November 15 , 1969 ( 1969 - 11 - 15 ) Mystery , Inc. meets a
strongman and a midget , both on the run from a circus allegedly haunted by a ghost clown . The gang goes to the circus grounds to investigate , but several of
them fall victim to the Ghost Clown 's powers of hypnotism .
</item 3>

<item 4>
List of Scooby-Doo, Where Are You! episodes
Wickles September 13 , 1969 ( 1969 - 09 - 13 ) While walking home from the movies , Shaggy and Scooby - Doo discover a black suit of armor in a pickup truck ,
and the gang deliver it to the local museum . The kids later break into the museum to search for clues , after learning that the archaeologist who was
transporting the suit is missing , and they find that the knight 's armor has come to life . `` Hassle in the Castle '' Phantom of Vasquez / Bluestone the Great
September 20 , 1969 ( 1969 - 09 - 20 ) The gang goes out on a boating expedition , but gets lost in a fog and crashes their boat on the mysterious Haunted Isle
. They decide to venture to the abandoned Vázquez Castle on the island for help , but instead encounter a transparent phantom running loose , who threatens them
to leave the island and never return .
</item 4>

<item 5>
List of Scooby-Doo, Where Are You! episodes
This is initially the series finale and the last episode under the Scooby - Doo , Where Are You ! name until September , 1978 . Season 3 ( 1978 ) ( revival ) (
edit ) Note : Only the first nine episodes of the sixteen episode 1978 season of Scooby - Doo , listed below , were broadcast under the Scooby - Doo , Where are
You ! name . The revival series was canceled afterward , and the remaining seven episodes , that were intended for the revival , were instead broadcast during
the Scooby 's All - Stars block , and all sixteen were rerun during this block and later syndicated as part of The Scooby - Doo Show . See List of The Scooby -
Doo Show episodes for the entire season run and The Scooby - Doo Show for proper production credits . No . overall No. in season Title Villain ( s ) Original
air date 26 `` Watch Out ! The Willawaw ! ''
</item 5>

<item 6>
List of Scooby-Doo, Where Are You! episodes
The Willawaw ! '' The Williwaw and Owl Men / Grey Fox and henchmen September 9 , 1978 ( 1978 - 09 - 09 ) While visiting Velma 's Uncle Dave , the gang find out
he has been kidnapped from his home , and that a mysterious Williwaw is haunting the area . 27 `` A Creepy Tangle in the Bermuda Triangle '' Skeleton Men / Dr.
Grimsley September 16 , 1978 ( 1978 - 09 - 16 ) While on a boat flowing down the Gulf Stream , the gang end up in the Bermuda Triangle and discover an airplane
- snatching flying saucer and a trio of Skeleton Men . 28 `` A Scary Night with a Snow Beast Fright '' Snow Beast / Dr. Bapiste September 23 , 1978 ( 1978 - 09
- 23 ) After being invited by Professor Krueger at the North Pole , the gang finds out that he has been captured . After they meet an Eskimo named Chief Minook
, he gets captured as well , and the gang attempts to solve the mystery .
</item 6>

<item 7>
List of Scooby-Doo, Where Are You! episodes
is a Saturday morning cartoon produced by Hanna - Barbera . Contents ( hide ) 1 Series overview 2 Episodes 2.1 Season 1 ( 1969 -- 70 ) 2.2 Season 2 ( 1970 ) 2.3
Season 3 ( 1978 ) ( Revival ) 3 References Series overview ( edit ) Season Episodes Originally aired First aired Last aired 17 September 13 , 1969 ( 1969 - 09 -
13 ) January 17 , 1970 ( 1970 - 01 - 17 ) 8 September 12 , 1970 ( 1970 - 09 - 12 ) October 31 , 1970 ( 1970 - 10 - 31 ) 16 September 9 , 1978 December 23 , 1978
Episodes ( edit ) Season 1 ( 1969 -- 70 ) ( edit ) No . overall No. in season Title Villain ( s ) Original air date `` What a Night for a Knight '' Black Knight
/ Mr.
</item 7>

<item 8>
List of Scooby-Doo, Where Are You! episodes
7 7 `` Never Ape an Ape Man '' Ape Man / Carl October 25 , 1969 ( 1969 - 10 - 25 ) The gang get jobs as extras on Daphne 's uncle John 's set of his new movie
The Ape Man of Forbidden Mountain , based on the legend of an ape man . However , the legend soon comes true and the ape man terrorizes the set and the gang . 8
8 `` Foul Play in Funland '' Charlie the Funland Robot controlled by Sarah Jenkins November 1 , 1969 ( 1969 - 11 - 01 ) The gang notice that the carnival
Funland appears to be operating by itself , so they go to investigate . They soon discover a crazy robot running loose , but when they notify the caretaker , he
and his wife are rebuffed , convincing the kids to solve the mystery . 9 9 `` The Backstage Rage '' Phantom Puppeteer / Mr.
</item 8>

<item 9>
List of Scooby-Doo, Where Are You! episodes
11 11 `` A Gaggle of Galloping Ghosts '' Count Dracula , Werewolf , Frankenstein 's Monster and the Gypsy / Big Bob Oakley , alias `` The Actor '' November 22 ,
1969 ( 1969 - 11 - 22 ) While on their way to Franken Castle , the gang stop to get their fortunes told by a gypsy . She warns them `` they will meet their doom
'' if they go to the castle . The gang goes to the castle anyway , where they are confronted by Count Dracula , Frankenstein 's monster and a Werewolf . 12 12
`` Scooby - Doo and a Mummy , Too '' Mummy of Ankha / Dr. Najib November 29 , 1969 ( 1969 - 11 - 29 ) While visiting a museum , the gang learn about the legend
of Ankha , a 3000 - year - old mummy . Legend had it that the mummy will come back alive and turn people to stone if he is removed from his tomb .
</item 9>

<item 10>
List of Scooby-Doo, Where Are You! episodes
21 `` Scooby 's Night with a Frozen Fright '' Caveman / Professor Wayne Seven Days a Week October 3 , 1970 ( 1970 - 10 - 03 ) While out fishing , Shaggy and
Scooby - Doo hook a two - million - year - old caveman frozen in a block of ice , which the gang learns was lost at sea during a violent storm . They take the
caveman to the Oceanland aquarium , but when someone melts the ice block and the caveman goes on the rampage , the gang decides to stick around and investigate
. 22 5 `` Haunted House Hang - Up '' Headless Spectre / Penrod Stillwall , Phantom / Asa Shanks Love the World October 10 , 1970 ( 1970 - 10 - 10 ) The Mystery
Machine overheats in front of an old spooky mansion , and the gang goes to look for help , but finds out that the house is being haunted by the Headless Spectre
, a terrifying phantom with no head . The mystery gets even stranger when they begin to hear knocking sounds and clues leading to a treasure .
</item 10>
=== END ITEM id=84ec9ee2323f26b2 ===
