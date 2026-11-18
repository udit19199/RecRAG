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

=== ITEM id=a2eb3b6be75db0f1 ===
Gold sentences and facts:
Sentence 0: William Alan `` Will '' Friedle ( born August 11 , 1976 ) is an American actor and voice actor . He is best known for his comedic roles , most notably the underachieving elder brother Eric Matthews on the long - running TV sitcom Boy Meets World from 1993 to 2000 . More recently , he has voiced a number of animated characters such as Terry McGinnis / Batman , the title character of Batman Beyond , and Ron Stoppable of Kim Possible . He voices Deadpool in Ultimate Spider - Man and Star - Lord in the animated version of Guardians of the Galaxy , replacing Chris Cox . He also performed the voices of Doyle in The Secret Saturdays , Lion - O in the rebooted ThunderCats series , and Blue Beetle on Batman : The Brave and the Bold . One of his most recent roles has been the speaking voice of Bumblebee in the final episode of Transformers : Prime and the movie Transformers Prime Beast Hunters : Predacons Rising and in the sequel series Transformers : Robots in Disguise , as well as in Transformers : Rescue Bots .
  Fact 0.0: William Alan "Will" Friedle was born on August 11, 1976.
  Fact 0.1: William Alan "Will" Friedle is an American actor.
  Fact 0.2: William Alan "Will" Friedle is an American voice actor.
  Fact 0.3: Will Friedle is best known for comedic roles.
  Fact 0.4: Will Friedle is most notably known for playing the underachieving elder brother Eric Matthews on the long-running TV sitcom Boy Meets World.
  Fact 0.5: Will Friedle played Eric Matthews on Boy Meets World from 1993 to 2000.
  Fact 0.6: Will Friedle has more recently voiced a number of animated characters.
  Fact 0.7: Will Friedle voiced Terry McGinnis / Batman, the title character of Batman Beyond.
  Fact 0.8: Will Friedle voiced Ron Stoppable of Kim Possible.
  Fact 0.9: Will Friedle voices Deadpool in Ultimate Spider-Man.
  Fact 0.10: Will Friedle voices Star-Lord in the animated version of Guardians of the Galaxy.
  Fact 0.11: Will Friedle replaced Chris Cox in the animated Guardians of the Galaxy role.
  Fact 0.12: Will Friedle performed the voice of Doyle in The Secret Saturdays.
  Fact 0.13: Will Friedle performed the voice of Lion-O in the rebooted ThunderCats series.
  Fact 0.14: Will Friedle performed the voice of Blue Beetle on Batman: The Brave and the Bold.
  Fact 0.15: One of Will Friedle's most recent roles was the speaking voice of Bumblebee in the final episode of Transformers: Prime.
  Fact 0.16: Will Friedle was the speaking voice of Bumblebee in the movie Transformers Prime Beast Hunters: Predacons Rising.
  Fact 0.17: Will Friedle was the speaking voice of Bumblebee in the sequel series Transformers: Robots in Disguise.
  Fact 0.18: Will Friedle was the speaking voice of Bumblebee in Transformers: Rescue Bots.

Retrieved items:
<item 1>
Daren Kagasoff
Daren Kagasoff - wikipedia Daren Kagasoff Jump to : navigation , search Daren Kagasoff At the Twilight premiere Daren Maxwell Kagasoff ( 1987 - 09 - 16 )
September 16 , 1987 ( age 30 ) Encino , California , U.S. Occupation Actor Years active 2008 -- present Daren Maxwell Kagasoff ( KA - guh - sawf ) ( born
September 16 , 1987 ) is an American actor . He is known for starring as Ricky Underwood on the ABC Family teen drama series The Secret Life of the American
Teenager from 2008 to 2013 . Contents ( hide ) 1 Early life 2 Career 3 Filmography 4 Awards and nominations 5 References 6 External links Early Life ( edit )
Kagasoff was born in Encino , California to Elise Kagasoff ( née Levy ) and Barry Kagasoff , a diamond merchant . He is the middle of three children ; he has an
older brother , Justin , and a younger sister , Natalie .
</item 1>

<item 2>
Will Friedle
Kiel ( 28 January 2009 ) . `` The Bravest & Boldest : Will Friedle '' . Comic Book Resources . Retrieved 30 November 2010 . Jump up ^ Andy Swift . `` ' Girl
Meets World ' : Eric Returns With Mr. Feeny -- Season 2 Spoilers - TVLine '' . TVLine . Retrieved 4 April 2016 . Jump up ^ `` Girl Meets Writers '' . Twitter .
Retrieved 4 April 2016 . Jump up ^ `` Will Friedle on Twitter '' . Twitter . Retrieved 4 April 2016 . Jump up ^ `` Art - World Movie Shot In State To Be Shown
Saturday In New Milford '' . tribunedigital - thecourant . Retrieved 4 April 2016 . Jump up ^ `` The Hartford Courant article archive - Friedle - Leary '' .
Retrieved 4 April 2016 . Jump up ^ `` The Hartford Courant article archive - She Chose to Have Both '' . Retrieved 4 April 2016 .
</item 2>

<item 3>
Daren Kagasoff
The film was released in October 2014 . In September 2014 , Kagasoff joined the cast of the Red Band Society , a comedy - drama series on Fox for the 2014 -- 15
American television season developed by Margaret Nagle . He had a recurring role playing Kara 's love interest , Hunter Cole . The series premiered on September
17 , 2014 and was canceled after one season . Filmography ( edit ) Television and film roles Year Title Role Notes 2008 -- 2013 Secret Life of the American
Teenager , The The Secret Life of the American Teenager Richard `` Ricky '' Underwood Main role 2014 Ouija Trevor Film 2014 Blue Daren Web series ; 2 episodes
2014 Delirium Alex Sheathes Unsold television pilot ( FOX ) 2014 Stalker Eric Bates Episode : Pilot 2014 -- 2015 Red Band Society Hunter Cole Recurring role , 7
episodes 2015 Paradise Pictures Wyatt Osborne Unsold television pilot ( USA ) 2017 S.W.A.T.
</item 3>

<item 4>
Daren Kagasoff (born September 16, 1987, in Encino, California) is an American actor. After playing baseball in high school and attending San Francisco State
University, he pursued acting. He is best known as Ricky Underwood in ABC Family’s *The Secret Life of the American Teenager* (2008–2013), a role that earned
him a 2009 Teen Choice Award. His other credits include the film *Ouija* (2014),
</item 4>

<item 5>
Will Friedle
Personal life ( edit ) In 1997 , Friedle dated American actress Jennifer Love Hewitt , and co-starred with her in the romantic teen comedy Trojan War . Hewitt
made a guest appearance as Jennifer Love `` Feffy '' Fefferman on Boy Meets World episode `` And Then There Was Shawn '' . Friedle is best friends with actor
Jason Marsden and was the best man at Marsden 's wedding in October 2004 . The two have also worked together in numerous productions : Boy Meets World , Trojan
War , Static Shock , Kim Possible , Batman Beyond , Justice League Unlimited , Batman : The Brave and the Bold , ThunderCats and Mad . He married his girlfriend
, Susan Martens , on September 25 , 2016 in Connecticut . Filmography ( edit ) Film Year Film Role Notes Trojan War Brad Kimble 2000 Batman Beyond : Return of
the Joker Terry McGinnis / Batman Voice role Direct - to - DVD release 2003 National Lampoon 's Gold Diggers Calvin Menhoffer The Greatest Short Film Ever ! ! !
</item 5>

<item 6>
Will Friedle
Career ( edit ) After co-hosting Nickelodeon 's Do n't Just Sit There for three seasons ( 1988 -- 1991 ) , Friedle was cast as Cory Matthews ' likeable but
underachieving elder brother Eric on the long - running TV sitcom Boy Meets World from 1993 to 2000 . In 2004 , he co-starred with Chris Owen in the independent
film National Lampoon 's Gold Diggers ( originally titled Lady Killers ) . He also appeared in a supporting role in the 2005 ABC Family movie Everything You
Want . Friedle was cast as Mike in the WB sitcom Off Centre , but could not accept the role because of a previous commitment to the UPN series The Random Years
. The role ended up going to Eddie Kaye Thomas . Friedle voiced Seifer Almasy in the English version of the video game Kingdom Hearts II , Gideon Wyeth in the
English version of the video game Advent Rising , and Jaster Rogue in the English version of the video game Rogue Galaxy .
</item 6>

<item 7>
Will Friedle
! ! Mark Short film Howl 's Moving Castle Additional voices Voice role 2005 Everything You Want Calvin Dillwaller 2008 Batman : Gotham Knight Anton , Cultist ,
Youth # 1 Voice role Direct - to - DVD release 2015 Batman Unlimited : Animal Instincts Nightwing / Dick Grayson Voice role Direct - to - DVD release Batman
Unlimited : Monster Mayhem Nightwing / Dick Grayson Voice role Direct - to - DVD release 2016 Lego DC Comics Super Heroes : Justice League : Gotham City
Breakout Nightwing / Dick Grayson Voice role Direct - to - DVD release Batman Unlimited : Mech vs. Mutants Nightwing / Dick Grayson Voice role Direct - to - DVD
release Television Year Title Role Notes Do n't Just Sit There Host Law & Order Russ Episode : `` Trust '' 1993 -- 2000 Boy Meets World Eric Matthews Series
regular ( 157 episodes ) Young Artist Award ( nominated ) The Gift of Love Luke CBS television movie Are You Afraid of the Dark ?
</item 7>

<item 8>
Will Friedle
Will Friedle - wikipedia Will Friedle Jump to : navigation , search Will Friedle William Alan Friedle ( 1976 - 08 - 11 ) August 11 , 1976 ( age 41 ) Hartford ,
Connecticut , U.S. Occupation Actor , voice actor Years active 1988 -- present Spouse ( s ) Susan Martens ( m . 2016 ) William Alan `` Will '' Friedle ( born
August 11 , 1976 ) is an American actor and voice actor . He is best known for his comedic roles , most notably the underachieving elder brother Eric Matthews
on the long - running TV sitcom Boy Meets World from 1993 to 2000 . More recently , he has voiced a number of animated characters such as Terry McGinnis /
Batman , the title character of Batman Beyond , and Ron Stoppable of Kim Possible . He voices Deadpool in Ultimate Spider - Man and Star - Lord in the animated
version of Guardians of the Galaxy , replacing Chris Cox . He also performed the voices of Doyle in The Secret Saturdays , Lion - O in the rebooted ThunderCats
series , and Blue Beetle on Batman : The Brave and the Bold .
</item 8>

<item 9>
Will Friedle (born 1976) is an American actor and voice actor, best known as Eric Matthews on *Boy Meets World* and for voicing Terry McGinnis/Batman in *Batman
Beyond*, Ron Stoppable in *Kim Possible*, and characters including Blue Beetle and Bumblebee. He reprised Eric on *Girl Meets World* and married Susan Martens
in 2016.

Harland Williams (born 1962) is a Canadian-American actor, comedian
</item 9>

<item 10>
Will Friedle
One of his most recent roles has been the speaking voice of Bumblebee in the final episode of Transformers : Prime and the movie Transformers Prime Beast
Hunters : Predacons Rising and in the sequel series Transformers : Robots in Disguise , as well as in Transformers : Rescue Bots . In December 2013 , it was
announced that Friedle would reprise his Boy Meets World role of Eric Matthews on the continuation series Girl Meets World . Contents ( hide ) 1 Early life 2
Career 3 Personal life 4 Filmography 5 References 6 External links Early life ( edit ) Friedle was born in Hartford , Connecticut , the son of Patricia Joan (
née Leary ) and Gary Allen Friedle , both lawyers . After committing himself to becoming an actor , Friedle continually commuted from Avon , Connecticut to New
York City for auditions . He attended Avon High School in 1994 . Friedle attended Occidental College . In the July 4 , 2006 issue of People , Friedle mentioned
he had thought about attending culinary school .
</item 10>
=== END ITEM id=a2eb3b6be75db0f1 ===

=== ITEM id=17d04d170bf5b054 ===
Gold sentences and facts:
Sentence 0: The Philadelphia Eagles are a professional American football franchise based in Philadelphia , Pennsylvania . The Eagles compete in the National Football League ( NFL ) as a member club of the league 's National Football Conference ( NFC ) East division . They are Super Bowl champions , having won Super Bowl LII , their fourth NFL title , after winning in 1948 , 1949 , and 1960 .
  Fact 0.0: The Philadelphia Eagles are a professional American football franchise.
  Fact 0.1: The Philadelphia Eagles are based in Philadelphia, Pennsylvania.
  Fact 0.2: The Philadelphia Eagles compete in the National Football League (NFL).
  Fact 0.3: The Philadelphia Eagles are a member club of the league's National Football Conference (NFC) East division.
  Fact 0.4: The Philadelphia Eagles are Super Bowl champions.
  Fact 0.5: The Philadelphia Eagles won Super Bowl LII.
  Fact 0.6: Super Bowl LII was the Eagles' fourth NFL title.
  Fact 0.7: The Eagles had won NFL titles in 1948, 1949, and 1960 before Super Bowl LII.
Sentence 1: Philadelphia Eagles Current season Established July 8 , 1933 ; 84 years ago ( July 8 , 1933 ) First season : 1933 Play in Lincoln Financial Field Philadelphia , Pennsylvania Headquartered in the NovaCare Complex Philadelphia , Pennsylvania Logo Wordmark League / conference affiliations National Football League ( 1933 -- present ) Eastern Division ( 1933 -- 1949 ) American Conference ( 1950 -- 1952 ) Eastern Conference ( 1953 -- 1969 ) Capitol Division ( 1967 -- 1969 ) National Football Conference ( 1970 -- present ) NFC East ( 1970 -- present ) Current uniform Team colors Midnight green , Silver , Black , White Fight song Fly , Eagles Fly Mascot Swoop Personnel Owner ( s ) Jeffrey Lurie Christina Weiss Lurie Chairman Jeffrey Lurie CEO Jeffrey Lurie President Don Smolenski General manager Howie Roseman Head coach Doug Pederson Team history Philadelphia Eagles ( 1933 -- 1942 ; 1944 -- present ) Phil - Pitt `` Steagles '' ( 1943 ) Team nicknames The Birds Championships League championships ( 4 ) NFL championships ( pre-1970 AFL -- NFL merger ) ( 3 ) 1948 , 1949 , 1960 Super Bowl championships ( 1 ) 2017 ( LII ) Conference championships ( 4 ) NFL Eastern : 1960 NFC : 1980 , 2004 , 2017 Division championships ( 13 ) NFL East : 1947 , 1948 , 1949 NFC East : 1980 , 1988 , 2001 , 2002 , 2003 , 2004 , 2006 , 2010 , 2013 , 2017 Playoff appearances ( 25 ) NFL : 1947 , 1948 , 1949 , 1960 , 1978 , 1979 , 1980 , 1981 , 1988 , 1989 , 1990 , 1992 , 1995 , 1996 , 2000 , 2001 , 2002 , 2003 , 2004 , 2006 , 2008 , 2009 , 2010 , 2013 , 2017 Home fields Baker Bowl ( 1933 -- 1935 ) Philadelphia Municipal Stadium ( 1936 -- 1939 , 1941 ) Connie Mack Stadium ( 1940 , 1942 -- 1957 ) Franklin Field ( 1958 -- 1970 ) Veterans Stadium ( 1971 -- 2002 ) Lincoln Financial Field ( 2003 -- present )
  Fact 1.0: The Philadelphia Eagles were established on July 8, 1933.
  Fact 1.1: The Eagles' first season was 1933.
  Fact 1.2: The Eagles play at Lincoln Financial Field in Philadelphia, Pennsylvania.
  Fact 1.3: The Eagles are headquartered in the NovaCare Complex in Philadelphia, Pennsylvania.
  Fact 1.4: The Eagles' team colors are midnight green, silver, black, and white.
  Fact 1.5: The Eagles' fight song is 'Fly, Eagles Fly'.
  Fact 1.6: The Eagles' mascot is Swoop.
  Fact 1.7: The Eagles are owned by Jeffrey Lurie and Christina Weiss Lurie.
  Fact 1.8: Jeffrey Lurie is the chairman of the Eagles.
  Fact 1.9: Jeffrey Lurie is the CEO of the Eagles.
  Fact 1.10: Don Smolenski is the president of the Eagles.
  Fact 1.11: Howie Roseman is the general manager of the Eagles.
  Fact 1.12: Doug Pederson is the head coach of the Eagles.
  Fact 1.13: The Eagles are nicknamed The Birds.
  Fact 1.14: The Eagles have been in the National Football League since 1933.
  Fact 1.15: The Eagles were in the Eastern Division from 1933 to 1949.
  Fact 1.16: The Eagles were in the American Conference from 1950 to 1952.
  Fact 1.17: The Eagles were in the Eastern Conference from 1953 to 1969.
  Fact 1.18: The Eagles were in the Capitol Division from 1967 to 1969.
  Fact 1.19: The Eagles have been in the National Football Conference since 1970.
  Fact 1.20: The Eagles have been in the NFC East since 1970.
  Fact 1.21: The Philadelphia Eagles team existed from 1933 to 1942 and from 1944 to the present.
  Fact 1.22: The Phil-Pitt 'Steagles' team existed in 1943.
  Fact 1.23: The Eagles have won 4 league championships.
  Fact 1.24: The Eagles have won 3 NFL championships before the AFL-NFL merger.
  Fact 1.25: The Eagles won NFL championships in 1948, 1949, and 1960.
  Fact 1.26: The Eagles have won 1 Super Bowl championship.
  Fact 1.27: The Eagles won Super Bowl LII in 2017.
  Fact 1.28: The Eagles have won 4 conference championships.
  Fact 1.29: The Eagles won the NFL Eastern conference championship in 1960.
  Fact 1.30: The Eagles won NFC conference championships in 1980, 2004, and 2017.
  Fact 1.31: The Eagles have won 13 division championships.
  Fact 1.32: The Eagles won NFL East division championships in 1947, 1948, and 1949.
  Fact 1.33: The Eagles won NFC East division championships in 1980, 1988, 2001, 2002, 2003, 2004, 2006, 2010, 2013, and 2017.
  Fact 1.34: The Eagles have made 25 playoff appearances.
  Fact 1.35: The Eagles made NFL playoff appearances in 1947, 1948, 1949, 1960, 1978, 1979, 1980, 1981, 1988, 1989, 1990, 1992, 1995, 1996, 2000, 2001, 2002, 2003, 2004, 2006, 2008, 2009, 2010, 2013, and 2017.
  Fact 1.36: The Eagles played home games at Baker Bowl from 1933 to 1935.
  Fact 1.37: The Eagles played home games at Philadelphia Municipal Stadium in 1936 to 1939 and in 1941.
  Fact 1.38: The Eagles played home games at Connie Mack Stadium in 1940 and from 1942 to 1957.
  Fact 1.39: The Eagles played home games at Franklin Field from 1958 to 1970.
  Fact 1.40: The Eagles played home games at Veterans Stadium from 1971 to 2002.
  Fact 1.41: The Eagles have played home games at Lincoln Financial Field since 2003.

Retrieved items:
<item 1>
Super Bowl XXXIX
( 1960 -- 1969 ) Seasons ( 58 ) 1960 1961 1962 1963 1964 1965 1966 1967 1968 1969 1970 1971 1972 1973 1976 1977 1978 1979 1980 1981 1982 1984 1985 1986 1988
1989 1990 1991 1992 1993 1994 1995 1997 1998 1999 2000 2001 2002 2003 2005 2006 2007 2008 2009 2011 2012 2013 2014 2015 2016 2017 2018 Championship seasons in
bold Philadelphia Eagles Founded in 1933 Based and headquartered in Philadelphia , Pennsylvania Franchise Franchise History Seasons Coaches Quarterbacks Players
Stadiums Baker Bowl Philadelphia Municipal Stadium Connie Mack Stadium Franklin Field Veterans Stadium Lincoln Financial Field Culture Fly , Eagles Fly Swoop
Curse of Billy Penn The Lombardi Curse Invincible Philadelphia Sports Hall of Fame Matt Guokas Sr . Dan Baker Cheerleaders
</item 1>

<item 2>
Super Bowl LI halftime show
Retrieved February 16 , 2017 . Jump up ^ Tuch , Paul ( February 13 , 2017 ) . `` On the Charts ... This Week '' . FYI Music Canada . Archived from the original
on February 14 , 2017 . Retrieved February 14 , 2017 . Jump up ^ Copsey , Rob ( February 7 , 2017 ) . `` Lady Gaga 's sales surge in the UK after Super Bowl
Halftime Show '' . Official Charts Company . Retrieved February 7 , 2017 . Jump up ^ Jones , Alan ( February 10 , 2017 ) . `` Official Charts Analysis : Elbow
debut at No. 1 with new album Little Fictions '' . Music Week . Archived from the original on February 13 , 2017 . Retrieved February 14 , 2017 . ( Subscription
required ( help ) ) . Jump up ^ Weatherby , Taylor ( February 5 , 2017 ) . `` Watch Lady Gaga 's Show - Stopping Super Bowl 2017 Halftime Performance '' .
Billboard .
</item 2>

<item 3>
Philadelphia Eagles
His rookie quarterback started with 5 touchdowns , no interceptions and over 255 yards per game . The Week 4 bye took a toll on the Eagles , and they lost four
out of the next five games , including a loss to every rival team in their division . They also lost right tackle Lane Johnson to a 10 - game suspension
following the Week 5 loss againist the Lions which damaged Carson Wentz 's hot start . In those four games , their average margin of loss was just under 5
points . Pederson and the Eagles won just 3 of their last 7 games . Those included gruesome road losses against the Seattle Seahawks , the Cincinnati Bengals ,
and the Baltimore Ravens . Although Wentz started off the season well , the 6 -- 5 quarterback finished with a TD -- INT ratio of 16 : 14 . The rookie head
coach and rookie quarterback tandem led the Eagles to a disappointing year , finishing with a 7 -- 9 record and coming in last in the division . 2017 : first
Super Bowl Championship During the following offseason , the team made several acquisitions on the offensive side of the ball .
</item 3>

<item 4>
Super Bowl XXXIX
Dan Baker Cheerleaders Silver Linings Playbook The Garbage Picking Field Goal Kicking Philadelphia Phenomenon Lore Frankford Yellow Jackets Pennsylvania
Keystoners `` Pennsylvania Polka '' Steagles `` Happy Hundred '' Miracle at the Meadowlands Buddy Ball Fog Bowl Bounty Bowl series Body Bag Game 4th and 26
Miracle at the New Meadowlands Philly Special Rivalries Dallas Cowboys New York Giants Washington Redskins Pittsburgh Steelers Division championships ( 13 )
1947 1948 1949 1980 1988 2001 2002 2003 2006 2013 2017 Conference championships ( 4 ) 1960 1980 2017 League championships ( 4 ) 1948 1949 1960 2017 ( LII )
Retired numbers 5 15 20 40 44 60 70 92 99 Media Broadcasters WTEL WIP - FM Merrill Reese Mike Quick Current league affiliations League : National Football
League Conference : National Football Conference Division : East Division Seasons ( 86 ) 1933 1934 1935 1936 1937 1938 1939 1940 1941 1942
</item 4>

<item 5>
Philadelphia Eagles
External links Wikimedia Commons has media related to Philadelphia Eagles . Official website Philadelphia Eagles Founded in 1933 Based and headquartered in
Philadelphia , Pennsylvania Franchise Franchise History Seasons Coaches Quarterbacks Players Stadiums Baker Bowl Philadelphia Municipal Stadium Connie Mack
Stadium Franklin Field Veterans Stadium Lincoln Financial Field Culture Fly , Eagles Fly Swoop Curse of Billy Penn The Lombardi Curse Invincible Philadelphia
Sports Hall of Fame Matt Guokas Sr . Dan Baker Cheerleaders Silver Linings Playbook The Garbage Picking Field Goal Kicking Philadelphia Phenomenon Lore
Frankford Yellow Jackets Pennsylvania Keystoners `` Pennsylvania Polka '' Steagles `` Happy Hundred '' Miracle at the Meadowlands Buddy Ball Fog Bowl Bounty
Bowl series Body Bag Game 4th and 26 Miracle at the New Meadowlands Philly Special Rivalries Dallas Cowboys New York Giants Washington Redskins Pittsburgh
Steelers Division championships ( 13 ) 1947 1948 1949 1980 1988 2001 2002 2003 2006 2013 2017 Conference championships ( 4 ) 1960 1980 2017 League championships
( 4 ) 1948 1949 1960 2017 ( LII ) Retired numbers 5
</item 5>

<item 6>
Philadelphia Eagles
2002 , 2003 , 2004 , 2006 , 2008 , 2009 , 2010 , 2013 , 2017 Home fields Baker Bowl ( 1933 -- 1935 ) Philadelphia Municipal Stadium ( 1936 -- 1939 , 1941 )
Connie Mack Stadium ( 1940 , 1942 -- 1957 ) Franklin Field ( 1958 -- 1970 ) Veterans Stadium ( 1971 -- 2002 ) Lincoln Financial Field ( 2003 -- present ) The
Philadelphia Eagles are a professional American football franchise based in Philadelphia , Pennsylvania . The Eagles compete in the National Football League (
NFL ) as a member club of the league 's National Football Conference ( NFC ) East division . They are Super Bowl champions , having won Super Bowl LII , their
fourth NFL title , after winning in 1948 , 1949 , and 1960 . The franchise was established in 1933 as a replacement for the bankrupt Frankford Yellow Jackets ,
when a group led by Bert Bell secured the rights to an NFL franchise in Philadelphia .
</item 6>

<item 7>
Doug Pederson
85 Terry Mickens 86 Antonio Freeman 87 Robert Brooks 88 Keith Jackson 89 Mark Chmura 90 Darius Holland 91 Shannon Clavelle 92 Reggie White 93 Gilbert Brown 94
Bob Kuberski 95 Keith McKenzie 96 Sean Jones 98 Gabe Wilkins Head coach : Mike Holmgren Coaches : Larry Brooks Nolan Cromwell Gil Haskell Johnny Holland Sherman
Lewis Jim Lind Tom Lovat Marty Mornhinweg Andy Reid Gary Reynolds Fritz Shurmur Harry Sydney Bob Valesente Philadelphia Eagles Super Bowl LII champions 4 Jake
Elliott 6 Caleb Sturgis 7 Nate Sudfeld 8 Donnie Jones 9 Nick Foles ( MVP ) 10 Mack Hollins 11 Carson Wentz 13 Nelson Agholor 14 Marcus Johnson 16 Bryce Treggs
17 Alshon Jeffery 18 Shelton Gibson 19 Rashard Davis 21 Patrick Robinson 22 Sidney Jones 23 Rodney McLeod 24 Corey Graham 26 Jaylen Watkins 27 Malcolm Jenkins
28 Wendell Smallwood 29 LeGarrette Blount 30 Corey Clement 31 Jalen Mills 32 Rasul Douglas 33 De'Vante Bausby 34 Donnel
</item 7>

<item 8>
Super Bowl XXXIX
For the Eagles , with the loss , the city of Philadelphia would not see its first championship since the 76ers swept the 1983 NBA Finals until the next time
their city 's teams played for a championship , when the Phillies won the 2008 World Series . The Eagles did not return to the Super Bowl until 2017 , also with
a 13 - 3 record , and defeating the Falcons and Vikings in the playoffs . The Eagles and the Patriots met again in Super Bowl LII , following the 2017 season ,
with the Eagles taking their revenge 41 -- 33 . This is the last time that a team has won back - to - back Super Bowls , with the Patriots losing their second
playoff game a year later and then , for the next eight years , every team either losing their first playoff game or missing them altogether . The streak was
broken in 2014 when the Seahawks defeated the Panthers and ultimately advanced to the Super Bowl , only to lose to a Patriots goal line stand .
</item 8>

<item 9>
Super Bowl XXXIX
The Eagles were trying to win their first NFL title since 1960 and the first championship for the city of Philadelphia since Moses Malone 's `` fo ' , fi ' , fo
' '' during the 76ers run to the 1983 NBA Championship . With President George W. Bush being inaugurated for a second time in January , the Eagles were also
trying to end a losing streak -- teams in the city of Philadelphia had lost six straight championships during presidential inauguration years , beginning with
the 76ers loss in 1977 . The streak included the Eagles in Super Bowl XV in 1981 . 1989 was not included in that streak , but 1977 , 1981 , 1985 ( Flyers ) ,
1993 ( Phillies ) , 1997 ( Flyers ) , and 2001 ( 76ers ) were . Television and entertainment ( edit ) The game was televised in the United States by Fox , with
play - by - play announcer Joe Buck and color commentators Cris Collinsworth and Troy Aikman . Pam Oliver ( Patriots sideline ) and Chris Myers ( Eagles
sideline ) served as sideline reporters .
</item 9>

<item 10>
Philadelphia Eagles
Championship Super Bowl champions quarterbacks Pro Bowl History League history Executive history Championship history Timeline Defunct franchises Franchise
moves and mergers Los Angeles team history Proposed stadiums 1995 -- 2016 American Football League ( 1960 -- 1969 ) Playoffs Merger NFL Championship ( 1920 --
1969 ) Playoff Bowl Records individual team Super Bowl All time win -- loss Last undefeated Tied games Canceled games Controversies Business Owners Properties
Management Council Competition Committee Collective Bargaining Agreement National Football League Players Association Lockouts Media TV NFL Network NFL RedZone
Radio NFL Films Other Officials Stadiums Chronology Awards All - Pro Hall of Fame Foreign players Player conduct Suspensions Player misconduct Combine Draft
Training camp Rivalries NFL Foundation Culture Cheerleading Color Rush Mascots Lore Nicknames Numbers Retired Sports teams based in Pennsylvania Baseball MLB
Philadelphia Phillies Pittsburgh Pirates IL Lehigh Valley IronPigs Scranton / Wilkes - Barre RailRiders EL Altoona Curve Erie SeaWolves Harrisburg Senators
Reading Fightin Phils NYPL State College Spikes Williamsport Crosscutters ALPB Lancaster Barnstormers York Revolution FL Washington Wild Things Basketball NBA
Philadelphia 76ers G League Erie BayHawks Cricket Lehigh Valley Cricket Club Philadelphia Cricket Club Football NFL Philadelphia Eagles Pittsburgh
</item 10>
=== END ITEM id=17d04d170bf5b054 ===

=== ITEM id=66c0ddbc592e73fe ===
Gold sentences and facts:
Sentence 0: R.H. Thomson as Matthew Cuthbert
  Fact 0.0: R.H. Thomson played the role of Matthew Cuthbert.

Retrieved items:
<item 1>
Anne (TV series)
Anne ( TV series ) - wikipedia Anne ( TV series ) Jump to : navigation , search Anne Also known as Anne with an E Genre Drama Created by Moira Walley - Beckett
Based on Anne of Green Gables by Lucy Maud Montgomery Screenplay by Moira Walley - Beckett Starring Amybeth McNulty Geraldine James R.H. Thomson Lucas Jade
Zumann Dalila Bela Corrine Koslo Aymeric Jett Montaz Opening theme `` Ahead by a Century '' by The Tragically Hip Country of origin Canada Original language ( s
) English No. of seasons No.
</item 1>

<item 2>
Anne (TV series)
Numeris . Jump up ^ Brioux , Bill ( @ BillBriouxTV ) ( April 10 , 2017 ) . `` SUN onites SNET NHL TOR - COL 1181k VAN -- EDM 766k TSN CURL 1026k GLO NCISLA 1Mk
CBC ANNE 654k STORY US 546k CITY SHOTS 623k CTV SAVE HOPE 548k '' ( Tweet ) -- via Twitter . Jump up ^ Brioux , Bill ( @ BillBriouxTV ) ( April 24 , 2017 ) . ``
SJN o'nites SNet NHL TOR - WASH Game 6 3467k OTT - BOS 1284k OMNI2 TOR - WASH 121k CTV SAVE HOPE 686K CBC ANNE 656K STORY OF US 450K '' ( Tweet ) -- via Twitter
. Jump up ^ `` Anne with an E ( 2017 - ) '' . Rotten Tomatoes . Retrieved 4 January 2018 . Jump up ^ `` Anne with an E : Season 1 '' . Metacritic .
</item 2>

<item 3>
Anne (TV series)
7 `` Wherever You Are Is My Home '' Amanda Tapping Moira Walley - Beckett April 30 , 2017 ( 2017 - 04 - 30 ) N / A The Cuthberts vow to do whatever it takes to
save the farm which reminds Anne of the strength of friendship and love . Reception ( edit ) Critical response ( edit ) The series has achieved a rating of 88 %
on Rotten Tomatoes , the site 's critical consensus stating : `` Anne with an E uses its complex central character to offer a boldly stylish , emotionally
resonant spin on classic source material that satisfies in its own right . '' The series has received a rating of 79 on Metacritic based on fifteen reviews ,
indicating `` generally favorable reviews '' . Emily Ashby , writing for Common Sense Media , calls the series an `` exceptional '' and `` spectacular ''
interpretation , giving it four out of five stars . Tasha Cerny , contributor for the Tracking Board , praises the cinematography as lush and colourful , the
characters vibrant , and the plot `` surprisingly thrilling for a story about a young girl living in a small secluded community in the late nineteenth century .
</item 3>

<item 4>
Anne (TV series)
The series premiered on March 19 , 2017 , on CBC , the season finale airing on April 30 , 2017 . On August 3 , 2017 , both CBC and Netflix renewed the series
for a 10 - episode second season to premiere in 2018 . Season 2 began production in November 2017 . Contents ( hide ) 1 Premise 2 Cast 2.1 Main 2.2 Others 3
Production 3.1 Personnel 3.2 Casting 3.3 Filming locations 3.4 Music 4 Themes 5 Broadcast and release 6 Episodes 6.1 Season 1 ( 2017 ) 7 Reception 7.1 Critical
response 8 References 9 External links Premise ( edit ) Brother and sister Matthew and Marilla Cuthbert , both in late middle age ( Matthew is 60 ) , decide to
take on an orphan boy to help out around their ancestral farm of Green Gables , on the outskirts of the town of Avonlea , on Prince Edward Island .
</item 4>

<item 5>
Anne (TV series)
When Matthew goes to pick the child up at the railway station , he finds not a boy , but a high - spirited and talkative girl , Anne Shirley . At first , the
Cuthberts are inclined to send her back , particularly after Marilla 's brooch goes missing , and Anne , in despair , runs away . Reconciled and settled in her
new home and school , Anne once again displays boundless enthusiasm which is nevertheless easily turned into despair when things go wrong , which they often do
. Slowly , her ebullient nature wins over those around her . Cast ( edit ) Main ( edit ) Amybeth McNulty as Anne Shirley Geraldine James as Marilla Cuthbert
R.H. Thomson as Matthew Cuthbert Dalila Bela as Diana Barry Lucas Jade Zumann as Gilbert Blythe Corrine Koslo as Rachel Lynde Aymeric Jett Montaz as Jerry
Baynard Others ( edit ) Jonathan Holmes as Mr. William Barry Helen Johns as Mrs. Eliza Barry Ryan Kiera Armstrong as Minnie Mary Barry Deborah Grover as
Josephine Barry Wayne Best as John Blythe Phillip Williams as Thomas Lynde David Ingram as Mr.
</item 5>

<item 6>
Anne (TV series)
of seasons No. of episodes 7 ( list of episodes ) Production Executive producer ( s ) Elizabeth Bradley Alex Sapot Sally Catto Debra Hayward Alison Owen Miranda
de Pencier Moira Walley - Beckett Producer ( s ) Susan Murdoch Cinematography Bobby Shore Running time 44 minutes Production company ( s ) Pelican Ballet
Northwood Entertainment Distributor Netflix Release Original network CBC Television Netflix ( worldwide ) Picture format 4K ( Ultra HD ) Audio format Dolby
Digital 5.1 with Descriptive Video Service track Original release March 19 , 2017 ( 2017 - 03 - 19 ) -- present ( present ) External links Website Anne ( titled
Anne with an E for international distribution on Netflix ) is a Canadian drama television series based on the 1908 novel Anne of Green Gables by Lucy Maud
Montgomery , and adapted by Emmy Award - winning writer and producer Moira Walley - Beckett . It airs on CBC in Canada , and elsewhere in the world it is
available for streaming on Netflix . The first season consists of seven episodes , with Niki Caro directing the 90 - minute season premiere .
</item 6>

<item 7>
Anne (TV series)
Writing for Entertainment Weekly , Isabella Beidenharn expressed similar feelings , but , `` putting the source material aside , it 's a fine show on its own ''
, and she conceded that `` inventing a dark side might help Anne With an E fit into today 's TV landscape '' . Allison Keene , writing for Collider , agrees
that Anne is a good drama on its own terms , but allows it is `` only a fair adaptation '' of the novel , at its best in the home scenes : `` Anne with an E is
undeniably the most stylish adaptation we 've ever seen of Anne of Green Gables . But its desire to reveal more of Anne 's miserable past in order to be more
true to what the desperation of an orphan is like feels at odds with Montgomery 's story . ''
</item 7>

<item 8>
Anne (TV series)
Jump up ^ `` Anne with an E : Season 1 '' . Metacritic . Retrieved 17 June 2017 . Jump up ^ Ashby , Emily . `` Anne with an E '' . Common Sense Media .
Retrieved 17 June 2017 . Jump up ^ Cerny , Tasha . `` ANNE WITH AN E Review : `` Episodes 1 - 3 '' `` . www.thetrackingboard.com . Retrieved 18 June 2017 . Jump
up ^ Inhat , Gwen . `` Anne With An E offers a winning , darker take on a familiar tale '' . A.V. Club . Retrieved 18 June 2017 . Jump up ^ Schneller , Johanna
( March 20 , 2017 ) . `` This Anne of Green Gables hints at miseries : What I 'm Watching '' . Toronto Star . access - date = requires url = ( help ) Jump up ^
Nguyen , Hanh . `` ' Anne with an E ' Review : A ' Breaking Bad ' Producer Updates ' Anne of Green Gables ' for Our Darker Times '' .
</item 8>

<item 9>
Anne (TV series)
'' She went on to call Anne Shirley an `` accidental feminist '' , and how she `` really wanted to tell this story now . '' Broadcast and Release ( edit ) The
series initially premiered on March 19 , 2017 , on CBC and aired on a weekly basis , the season finale airing on April 30 , 2017 . The series debuted on Netflix
on May 12 , 2017 , under the title Anne with an E . Episodes ( edit ) Season 1 ( 2017 ) ( edit ) No . Title Directed by Written by Original air date Can .
viewers ( millions ) `` Your Will Shall Decide Your Destiny '' Niki Caro Moira Walley - Beckett March 19 , 2017 ( 2017 - 03 - 19 ) 0.999 When a miscommunication
brings a girl , Anne Shirley , to Green Gables instead of a boy , Marilla and Matthew Cuthbert are faced with a life - changing decision .
</item 9>

<item 10>
Anne (TV series)
IndieWire . Retrieved 17 June 2017 . Jump up ^ Chaney , Jen . `` Anne of Green Gables Fans , You Will Love Netflix 's Anne With an E '' . Vulture.com . New York
Magazine . Retrieved 17 June 2017 . Jump up ^ Genzlinger , Neil ( May 11 , 2017 ) . `` Review : ' Anne With an E ' Is a Rewarding Return to Green Gables '' .
The New York Times . Retrieved 17 June 2017 . Jump up ^ Hirschlag , Annie . `` ' Anne with an E ' Review : The Neflix Adpatation Hits the Nostalgia Button and
More '' . Mic.com . Mic ( media Company ) . Retrieved 17 June 2017 . Jump up ^ Nawaz , Saleema ( March 21 , 2017 ) . `` We asked an Anne of Green Gables
superfan to review the first episode of CBC 's new TV series '' . Toronto Life . Retrieved 4 June 2017 .
</item 10>
=== END ITEM id=66c0ddbc592e73fe ===

=== ITEM id=e0b5eae77f49be09 ===
Gold sentences and facts:
Sentence 0: No . overall No. in season Title Directed by Written by Original air date U.S. viewers ( millions ) 112 `` The Savior '' Eagle Egilsson Edward Kitsis & Adam Horowitz September 25 , 2016 ( 2016 - 09 - 25 ) 3.99 In the past , Aladdin , a `` Savior '' , is taunted by Jafar over the inevitable downfall of all Saviors . In Storybrooke , refugees arrive from the Land of Untold Stories . Emma experiences tremors and visions ; an arrested Hyde helps her receive an oracle 's vision of a hooded figure killing her . Guided by Morpheus , Gold enters Belle 's dreams to lift the sleeping curse ; he tries to renew their love , but she rejects him . `` Morpheus '' reveals he is really their unborn son , and desired this outcome . Belle wakes and leaves Gold . Henry convinces Regina that Hades may have lied about Robin Hood 's soul having been destroyed . Zelena moves in with Regina but , after learning her sister blames her for Robin Hood 's death , moves out and is approached by the Evil Queen . 113 `` A Bitter Draught '' Ron Underwood Andrew Chambliss & Dana Horgan October 2 , 2016 ( 2016 - 10 - 02 ) 3.72 In the past in the Enchanted Forest , the Evil Queen offers to aid the revenge plot of Edmond Dantes , The Count of Monte Cristo , if he kills Snow and David . When he falls in love with Snow 's handmaiden , he reneges and flees with her to the Land of Untold Stories to save her from poison . In Storybrooke , the Evil Queen recruits Zelena and uses Edmond 's heart to set him against Snow and David ; Regina kills Edmond to save the others , and realizes she retains the capacity for darkness . The Evil Queen declares that the undesirable `` untold stories '' will be played out . She gives David evidence that his father 's death was not accidental . Belle lives on Hook 's ship ; in a deal with Gold , the Evil Queen will not harm her or the baby . Emma tells Archie about her visions and notes that Regina is n't present in them and could be the hooded figure . 114 `` The Other Shoe '' Steve Pearlman Jane Espenson & Jerome Schwartz October 9 , 2016 ( 2016 - 10 - 09 ) 4.11 In the past in the Enchanted Forest , Cinderella meets Prince Thomas at the ball , but flees when she believes he loves her stepsister Clorinda ; Clorinda actually plans to elope with Thomas ' footman , Jacob , but her mother , Lady Tremaine , abducts her to the Land of Untold Stories after forcing Cinderella to betray Clorinda 's location . In Storybrooke , Ashley hopes to reconcile with Clorinda . Aided by the Evil Queen , Tremaine lures Ashley into a trap for her and Jacob . Ashley protects Jacob ; Clorinda reunites with him and forgives Cinderella , while Tremaine is arrested . Emma struggles to control her magic . Whale helps Jekyll recreate his serum , while the Evil Queen and Hyde form an alliance and she releases him . Snow White wants normality , but David secretly investigates his father 's death ; Gold provides information in exchange for David delivering a message to Belle . 115 `` Strange Case '' Alrick Riley David H. Goodman & Nelson Soler October 16 , 2016 ( 2016 - 10 - 16 ) 3.53 In the past in Victorian England , Jekyll petitions the Academy of Science for membership but Dr. Lydgate rejects him . Rumplestiltskin arrives and magically perfects Jekyll 's serum ; Jekyll transforms into Hyde , who blackmails Lydgate for Jekyll 's membership . Mary is Lydgate 's daughter and Jekyll 's unrequited love ; she and Hyde fall in love . Jekyll accidentally fatally defenestrates Mary ; he frames Hyde . In Storybrooke , Gold imprisons Belle on the ship to protect her from Hyde ; but it is Jekyll who tries to kill Belle in revenge for Rumplestiltskin 's role in Mary 's death . Hook saves Belle by killing Jekyll , and Hyde dies of identical wounds ; Regina asks Emma to kill her if necessary to stop the Evil Queen . Gold warns Belle that their child will need his protection . Mary Margaret resumes teaching , and her assistant Shirin inspires her to draw on her life as Snow . Shirin is secretly Princess Jasmine , working with the Oracle to search for Aladdin . 116 5 `` Street Rats '' Norman Buckley Edward Kitsis & Adam Horowitz October 23 , 2016 ( 2016 - 10 - 23 ) 3.40 In the past in Agrabah , Aladdin seeks a magical weapon for Jasmine , to save the kingdom from Jafar ; he is the weapon and a Savior . Jafar tempts Aladdin with magical shears that can sever him from his fate as the Savior and from his resulting death . Aladdin rejects Jafar and saves Jasmine . In Storybrooke , the Oracle is murdered . Emma helps Jasmine , hoping Aladdin 's survival ensures her own . Having faked his death , Aladdin reveals himself to Emma and confesses that he doomed Agrabah by using the shears , which he gives to her ; she asks Hook to dispose of them , but he secretly keeps them . The Evil Queen impersonates Archie to learn of Emma 's visions , and convinces Zelena to turn Archie back into a cricket . 117 6 `` Dark Waters '' Robert Duncan Andrew Chambliss & Brigitte Hales October 30 , 2016 ( 2016 - 10 - 30 ) 3.06 During the Dark Curse , Captain Nemo abducts Hook and invites him to join the crew `` family '' of his submarine , Nautilus , in their quest for the `` Mysterious Island '' , which is the Land of Untold Stories . Hook discovers the first mate is his half - brother , Liam , and decides to leave to avoid confrontation ; but Liam attacks him , instead mortally wounding Nemo , who steps in . In the present , Archie is freed . Henry confronts Hook over lying about the shears , and Liam imprisons them on Nautilus . Hook opens up and Henry accepts him as family ; they save each other and dispose of the shears . Modern medicine saves Nemo ; he , Liam , and Hook make amends . Emma and Aladdin discuss being a Savior . Jasmine reveals that Agrabah vanished . Snow supports Belle at her ultrasound . The Evil Queen seduces Gold and offers the retrieved shears in exchange for help taking Snow 's heart . 118 7 `` Heartless '' Ralph Hemecker Jane Espenson November 6 , 2016 ( 2016 - 11 - 06 ) 3.56 In the past , fugitive Snow and shepherd David separately travel to Longbourn ; she to flee the kingdom and he to sell his struggling farm . Without ever seeing each other 's faces , David saves Snow from a bounty hunter and she repays him with money to save the farm , deciding not to flee . Their hands touch and a magical sapling is created . In the present , the Evil Queen threatens to destroy Storybrooke with River of Lost Souls water unless Mary Margaret and David hand over their hearts . They learn the sapling could stop the Queen , who destroys it just after they find it . But it has shown them their previous meeting , and they conclude that their love can endure anything ; they surrender themselves . The Queen casts a sleeping curse on their hearts ; because they share Snow 's heart , the curse affects them both ; each can wake the other but immediately succumbs to the curse , so they can never be awake together . Regina uses Gold 's relationship with the Queen to separate the latter from Zelena , who earns Gold 's enmity by exposing his schemes to Belle . 119 8 `` I 'll Be Your Mirror '' Jennifer Lynch Jerome Schwartz & Leah Fong November 13 , 2016 ( 2016 - 11 - 13 ) 3.40 Regina and Emma try to magically trap the Evil Queen in the World Behind the Mirror , but she traps them there , where the Dragon is already imprisoned . Henry prepares for a date with Violet ; the Queen impersonates Regina and tries to manipulate him but he recognizes her . Emma and Regina discover Sidney was building a mirror portal out of the world and try and finish it , but the Evil Queen forces the Dragon to attack Emma and Regina , using his heart , and tries to make Henry kill the Dragon with the Hammer of Hephaestus ; he instead uses it to free his mothers . Mary Margaret and David adjust to living in shifts . Belle asks Zelena to help her escape Storybrooke via magical portal , and they form an alliance with Aladdin and Jasmine . Aladdin steals the Genie 's lamp and the Sorcerer 's wand from Gold 's shop , but Gold takes the wand back from Zelena and puts a magical tracking device on Belle . He asks the Queen to kill Zelena , as the connection between his and Zelena 's hearts protects her from him . 120 9 `` Changelings '' Mairzee Almas David H. Goodman & Brian Ridings November 27 , 2016 ( 2016 - 11 - 27 ) 3.28 In the Enchanted Forest , while served by Belle , Rumplestiltskin abducts a baby to use as bait for the Black Fairy , the mother who abandoned him ; she escapes without answering his questions . In the present , Belle receives advice from her unborn child while Gold becomes increasingly unhinged . Resenting Gold 's order to attack Zelena , the Evil Queen accelerates Belle 's pregnancy ; Mother Superior delivers Belle 's son , Gideon , and sends him into hiding ; Gold vows to find him . As the Genie has been freed , Aladdin takes his place . Zelena and Regina continue to feud . Emma obtains the sword from her vision . 121 10 `` Wish You Were Here '' Ron Underwood Edward Kitsis & Adam Horowitz December 4 , 2016 ( 2016 - 12 - 04 ) 3.27 The Black Fairy abducts Gideon . Emma discovers the sword can kill the Evil Queen without hurting Regina . The Queen steals the lamp and wishes Emma to an alternate reality where she was never the Savior ; Regina follows her . David and his allies take the lamp from the Queen , who is turned into a serpent by the hooded figure , revealed to be the adult Gideon . Jasmine wishes herself and Aladdin to Agrabah . In the `` wish realm '' , Emma lives as a princess until Regina traumatically restores her memory . Regina frees Rumplestiltskin in exchange for a magic bean , but the portal to Storybrooke closes while Regina is distracted by the appearance of Robin Hood . 122 11 `` Tougher Than the Rest '' Billy Gierhart Edward Kitsis & Adam Horowitz March 5 , 2017 ( 2017 - 03 - 05 ) 3.03 In 1990 Minnesota , a young Emma is told by a teenager that she has the power to change her fate . In present day , after they miss their way out of the Wish realm , Emma and Regina find another option for escape when they come across August , who agrees to help create a new portal . Regina goes to find Robin Hood to see if he would 've been better off without her . After Regina finds Robin , they 're caught by the Sheriff of Nottingham and later by the Wish realm Rumple . Robin believes Regina when she tells him about their being in an alternate reality . Emma discovers that August was the teenager she met back in Minnesota and inspires him to complete the portal . Emma , Regina , and Robin return to Storybrooke , where Gideon explains to Gold and Belle his intention to kill Emma so that he can acquire her savior powers and kill the Black Fairy , a move that has David furious and Belle concerned . Emma and Gideon finally meet for the battle , but Emma survives , choosing her own fate . Gideon retreats , proclaiming this far from over . Gold and Belle decide to work together for the good of their son . 123 12 `` Murder Most Foul '' Morgan Beggs Jerome Schwartz & Jane Espenson March 12 , 2017 ( 2017 - 03 - 12 ) 3.06 The events of how Robert , the father of David and James , was murdered are detailed in a flashback , in which Robert learns what happened to James after he ran away from King George . With help from Rumple , he tracks him to Pleasure Island , but their reunion is cut short by King George , who orders Robert 's execution . His death affects David in the present day after he sees visions of Robert and asks Hook to help him , despite his being reformed . When Hook becomes concerned that David is taking this investigation too far by going after Albert , he intervenes and brings David back to sanity . Hook is also worried about whether David will give him his blessing to marry Emma , but David does . Meanwhile , Regina tries to get the Wish Realm Robin assimilated into Storybrooke , but soon questions why he agreed to follow her , as it turns out that he had something else planned after he steals a chest from the vault . The truth as to who really murdered Robert is revealed when August gives Hook missing pages he took from Henry 's book . When Hook sees an illustration of David 's dad , he recognizes him and recalls murdering him during a robbery in the Enchanted Forest so that there was no witness to his crime . 124 13 `` Ill - Boding Patterns '' Ron Underwood Andrew Chambliss & Dana Horgan March 19 , 2017 ( 2017 - 03 - 19 ) 2.71 During the First Ogres War , Beowulf leads a fearsome charge on the battlefield in the Enchanted Forest of old . After every last soldier had been slain , Beowulf surrenders to death just as Rumplestiltskin intervenes , killing all the ogres . Hailed as a hero in the village , Beowulf is irked by Rumple 's new title . He creates a ruse to lead Rumple to his demise . Fearing he would not be able to follow through on protecting the villagers without Dark Magic , Rumple bestows his dagger upon Baelfire as an insurance policy . Following a turbulent exchange and threat to frame Rumplestiltskin as a murderer , Beowulf retreats . Feeling the pressure of his threat , Bae commands his father to kill him . Rumple later doses Bae with a memory potion to erase the ordeal from his mind and restore his innocence . In Storybrooke , Rumple recognizes a familiar pattern in Gideon , who makes strides in his quest to kill Emma Swan at the cost of The Blue Fairy 's vitality . Hook worries about the repercussions of telling Emma his newly unveiled secret . Meanwhile , Emma unknowingly forces a proposal after happening upon Hook 's engagement ring . Robin of Locksley attempts to flee the town , but to no avail . The Evil Queen escapes her reptilian confinement and indoctrinates Robin . 125 14 `` Page 23 '' Kate Woods David H. Goodman & Brigitte Hales March 26 , 2017 ( 2017 - 03 - 26 ) 2.85 In the Enchanted Forest , The Evil Queen is in search of a spell that will help her end Snow White for good , but her father has love in mind for his daughter and leads her on a quest to Cupid 's Arrow . However , The Evil Queen rebrands the arrow to follow hatred and discovers that the person she despises the most is herself . In Storybrooke , The Evil Queen and Regina face off in a final confrontation . Just when Regina wins the upper hand , she remembers the Evil Queen 's self - loating and evenly splits the light and the darkness in their hearts , to share their love and hate and allow the Evil Queen to reform . After Emma calls off the engagement , Hook decides to board the Nautilus with Captain Nemo . Hook later changes his mind following a conversation with Snow but , before he can return , Gideon hijacks the vessel and sends it out of Storybrooke . 126 15 `` A Wondrous Place '' Steve Pearlman Jane Espenson & Jerome Schwartz April 2 , 2017 ( 2017 - 04 - 02 ) 2.80 At a loss for hope , Regina suggests a `` girls ' night out '' for Emma with her and Snow . While at the bar , she confides in the mixologist and sheds a tear . Meanwhile , the Nautilus is transported to the Enchanted Forest , where Aladdin and Jasmine board . They have yet to locate Agrabah and are growing weary . After Hook and Jasmine confide in one another , the vessel is irrevocably damaged and Jasmine uses a genie wish to relocate the group to the island where Jafar is located . They discover a cabana belonging to Ariel and Prince Eric . Ariel has collected a genie lamp that houses Jafar . Jasmine is able to sway him into divulging the location of the hidden Agrabah before defeating him with magic dust . Through true love 's kiss , Aladdin and Jasmine , along with Hook and Ariel , are transported to the restored city and Aladdin 's genie curse is broken . Ariel supplies Hook with a means to reach Emma . Emma receives the message and Gideon , no longer masquerading as the mixologist , threatens that he will use her tears to keep Hook from returning to Storybrooke unless she helps him kill the Black Fairy . Flashbacks to Ariel and Jasmine 's first meeting show the princess falter under Jafar 's trickery . 127 16 `` Mother 's Little Helper '' Billy Gierhart Story by : Edward Kitsis & Adam Horowitz Teleplay by : Paul Karp April 9 , 2017 ( 2017 - 04 - 09 ) 2.60 In Storybrooke , Gold reveals to Emma and Snow that the Black Fairy is his mother and that the Dark Curse Regina cast was created by the Black Fairy . Emma agrees to help Gideon kill the Black Fairy , but is double - crossed by Gideon and left to die at the hands of a giant spider . Gideon opens a portal with the sword , allowing the Black Fairy to slip through into Storybrooke . Meanwhile in the Enchanted Forest , Hook wagers the Jolly Roger for a magic bean in a game of cards with Blackbeard . As Blackbeard and Hook go through the portal , it malfunctions thanks to Gideon and the duo ends up in Neverland , on the run from the remnants of Peter Pan 's Lost Boys . Back in Storybrooke , Henry experiences a disorienting shift in his Author powers , so he and Regina seek help from the only person who can aid them : the previous Author , Isaac . Isaac reveals to Regina and Henry that they are nearing the end of the story in which the savior must fight the Final Battle . Flashbacks reveal the abuses the Black Fairy inflicted on Gideon and that the Black Fairy ripped out Gideon 's heart and is controlling him to kill Emma . 128 17 `` Awake '' Sharat Raju Andrew Chambliss & Leah Fong April 16 , 2017 ( 2017 - 04 - 16 ) 2.51 Regina works feverishly to wake Snow and Charming from the Sleeping Curse . Gold comes face to face with the Black Fairy , the powerful enemy Emma must fight in the Final Battle . Hook , still in Neverland , meets up with Tiger Lily , who gives him a piece of an ancient fairy wand to aid Emma in defeating the Black Fairy . Gold confronts the Black Fairy over her manipulating Gideon with his heart and she maintains that Storybrooke would be destroyed if the two of them fought . A pixie flower allows Emma to open a doorway to Neverland and rescue Hook , who then re-proposes properly . The townsfolk gather around Emma 's sleeping parents and each take small amounts of the Sleeping Curse unto themselves , diluting it and allowing Snow and David to finally reunite . In 1993 , a pixie flower awakens Snow and Charming from their cursed selves , but an awakened Rumplestiltskin tells them they must make a difficult decision to ensure that Emma fulfills her destiny as the Savior . 129 18 `` Where Bluebirds Fly '' Michael Schultz David H. Goodman & Brigitte Hales April 23 , 2017 ( 2017 - 04 - 23 ) 2.69 In Zelena 's past , her childhood friend Stanum asks for her help in finding the Crimson Heart , a magical artifact that would be able to restore his human form after having been cursed by the Wicked Witch of the North to slowly turn into tin . Zelena 's refusal to give up her magic leads to her betraying Stanum , as she keeps the Heart and leaves him to become the Tin Man . In the Present , The Black Fairy tempts Zelena with an offer to join her , only to have Zelena oppose her instead . Regina is brought into the chaos by trying to stop her sister and they find the Black Fairy and Gideon in the crystal mines , with the Black Fairy tricking Zelena into utilizing her unstable Dark Magic to strengthen her for the Final Battle . Realizing that there is nothing left for her back in Oz , Zelena decides to help the residents fight back against the Black Fairy by using the Crimson Heart to give up her magic , thereby undoing the mess she made in the mines . Emma then uses her magic and one of the crystals to bring Mother Superior out of her coma with some help from Regina and Zelena . 130 19 `` The Black Fairy '' Alrick Riley Jerome Schwartz & Dana Horgan April 30 , 2017 ( 2017 - 04 - 30 ) 3.05 The origin of Rumplestiltskin 's true destiny is revealed , as his mother Fiona learns from the Blue Fairy and Tiger Lily that he was destined to become The Savior . When Fiona fears for her son and chooses power over protecting Rumple , it results in her becoming the Black Fairy , and the Blue Fairy banishing her to the Dark Realm . In the Present , as everyone looks for the missing wand , Gold brings Gideon and Emma into the Dream World to search for Gideon 's heart , where the events of the flashbacks unfold . Gold then confronts Fiona , but she reveals the real reason behind why he was abandoned . Gold later retrieves Gideon 's heart and tells everyone that the Black Fairy is dead , paving the way for Emma and Hook 's wedding . However , Gold and Fiona make a deal that will see the Final Battle go forward , with Emma being the Savior that will fight in it . Meanwhile , Regina helps Zelena adjust to life without her powers . 131 20 `` The Song in Your Heart '' Ron Underwood David H. Goodman & Andrew Chambliss May 7 , 2017 ( 2017 - 05 - 07 ) 2.87 In the past , Snow makes a wish that grants the entire Enchanted Forest the ability to use the gift of song as a strength , which also spreads to Oz as well . When Regina becomes affected by the wish , she finds a way to end it before the first curse takes place , but the Blue Fairy , who granted Snow the wish , tells Snow and David that the wish will be part of Emma 's destiny . In Emma 's past , her chance to make use of her talent is held back by being told she 'll be alone . This would come back to taunt Emma in the present day , as Fiona returns to use that weakness to take Emma 's heart before unleashing a new curse on her wedding day , with help from Gold by freezing her family . When Henry discovers the page in the book that reveals the song in Emma 's heart will make her stronger , not only does it embolden Emma , it stops Fiona from crushing her heart , but Fiona still tells Emma that the battle will continue . On her wedding day , Emma and Hook finally marry , just in time for the dark clouds of the Black Fairy 's curse to erupt from the clock tower and engulf them at 6 : 00 PM , setting up the final battle . 132 133 21 22 `` The Final Battle '' Steve Pearlman Ralph Hemecker Edward Kitsis & Adam Horowitz May 14 , 2017 ( 2017 - 05 - 14 ) 2.95 Fiona 's curse alters Storybrooke , institutionalizing Emma and leaving only Henry and Gold `` awake , '' while sending Emma 's other allies to their home realms . The Final Battle begins over Emma 's belief ; if she stops believing in the realms of story , they will be consumed by the Darkness and cease to exist . If all the realms are destroyed , Fiona would gain unchecked powers and be able to break the Laws of Magic at will . In the Enchanted Forest , the banished characters gather at Regina 's castle ; David and Hook obtain a magic bean and the Evil Queen sacrifices herself , holding off the Darkness to delay the group 's destruction . Fiona convinces Emma to destroy the book , but Henry hand - draws one that restores her belief in time to save everyone in the castle . Gold kills Fiona , who had hidden Belle and turned her into a coward ; this breaks the Curse , which restores everyone 's memories and reunites them in Storybrooke . However , Gideon is still under orders to kill Emma . Gold attempts to free Gideon 's heart ; however , a spell cast by the Black Fairy prevents him from doing so . Emma lays down her sword , allowing Gideon to impale her ; this releases Emma 's light magic and ends the battle . With both Gold and Emma having chosen good , Gideon is returned to infancy and the realms are restored , and those who died in the crisis are revived . The characters embark on their `` happy beginnings . '' Many years later in the New Enchanted Forest , an adult Henry vanishes while defending his daughter Lucy from an unidentified entity . Guided by Tiger Lily , Lucy travels to Seattle , where Henry no longer remembers her .
  Fact 0.0: Episode 112, 'The Savior', was directed by Eagle Egilsson and written by Edward Kitsis and Adam Horowitz.
  Fact 0.1: Episode 112 aired on September 25, 2016 and had 3.99 million U.S. viewers.
  Fact 0.2: In the past, Aladdin, a 'Savior', is taunted by Jafar over the inevitable downfall of all Saviors.
  Fact 0.3: In Storybrooke, refugees arrive from the Land of Untold Stories.
  Fact 0.4: Emma experiences tremors and visions.
  Fact 0.5: An arrested Hyde helps Emma receive an oracle's vision of a hooded figure killing her.
  Fact 0.6: Guided by Morpheus, Gold enters Belle's dreams to lift the sleeping curse.
  Fact 0.7: Gold tries to renew his love with Belle, but she rejects him.
  Fact 0.8: 'Morpheus' reveals he is really Belle and Gold's unborn son, and that he desired this outcome.
  Fact 0.9: Belle wakes and leaves Gold.
  Fact 0.10: Henry convinces Regina that Hades may have lied about Robin Hood's soul having been destroyed.
  Fact 0.11: Zelena moves in with Regina.
  Fact 0.12: After learning her sister Regina blames her for Robin Hood's death, Zelena moves out.
  Fact 0.13: Zelena is approached by the Evil Queen after moving out.
  Fact 0.14: Episode 113, 'A Bitter Draught', was directed by Ron Underwood and written by Andrew Chambliss and Dana Horgan.
  Fact 0.15: Episode 113 aired on October 2, 2016 and had 3.72 million U.S. viewers.
  Fact 0.16: In the past in the Enchanted Forest, the Evil Queen offers to aid the revenge plot of Edmond Dantes, The Count of Monte Cristo.
  Fact 0.17: The Evil Queen's offer to Edmond Dantes is conditional on him killing Snow and David.
  Fact 0.18: Edmond Dantes falls in love with Snow's handmaiden.
  Fact 0.19: Edmond Dantes reneges on the revenge plot and flees with the handmaiden to the Land of Untold Stories to save her from poison.
  Fact 0.20: In Storybrooke, the Evil Queen recruits Zelena.
  Fact 0.21: In Storybrooke, the Evil Queen uses Edmond's heart to set him against Snow and David.
  Fact 0.22: Regina kills Edmond to save the others.
  Fact 0.23: Regina realizes she retains the capacity for darkness.
  Fact 0.24: The Evil Queen declares that the undesirable 'untold stories' will be played out.
  Fact 0.25: The Evil Queen gives David evidence that his father's death was not accidental.
  Fact 0.26: Belle lives on Hook's ship.
  Fact 0.27: In a deal with Gold, the Evil Queen will not harm Belle or the baby.
  Fact 0.28: Emma tells Archie about her visions.
  Fact 0.29: Emma notes that Regina is not present in her visions and could be the hooded figure.
  Fact 0.30: Episode 114, 'The Other Shoe', was directed by Steve Pearlman and written by Jane Espenson and Jerome Schwartz.
  Fact 0.31: Episode 114 aired on October 9, 2016 and had 4.11 million U.S. viewers.
  Fact 0.32: In the past in the Enchanted Forest, Cinderella meets Prince Thomas at the ball.
  Fact 0.33: Cinderella flees when she believes Thomas loves her stepsister Clorinda.
  Fact 0.34: Clorinda actually plans to elope with Thomas' footman, Jacob.
  Fact 0.35: Clorinda's mother, Lady Tremaine, abducts Clorinda to the Land of Untold Stories.
  Fact 0.36: Lady Tremaine abducts Clorinda after forcing Cinderella to betray Clorinda's location.
  Fact 0.37: In Storybrooke, Ashley hopes to reconcile with Clorinda.
  Fact 0.38: Aided by the Evil Queen, Lady Tremaine lures Ashley into a trap for her and Jacob.
  Fact 0.39: Ashley protects Jacob.
  Fact 0.40: Clorinda reunites with Jacob and forgives Cinderella.
  Fact 0.41: Lady Tremaine is arrested.
  Fact 0.42: Emma struggles to control her magic.
  Fact 0.43: Whale helps Jekyll recreate his serum.
  Fact 0.44: The Evil Queen and Hyde form an alliance, and she releases him.
  Fact 0.45: Snow White wants normality.
  Fact 0.46: David secretly investigates his father's death.
  Fact 0.47: Gold provides information in exchange for David delivering a message to Belle.
  Fact 0.48: Episode 115, 'Strange Case', was directed by Alrick Riley and written by David H. Goodman and Nelson Soler.
  Fact 0.49: Episode 115 aired on October 16, 2016 and had 3.53 million U.S. viewers.
  Fact 0.50: In the past in Victorian England, Jekyll petitions the Academy of Science for membership.
  Fact 0.51: Dr. Lydgate rejects Jekyll's petition.
  Fact 0.52: Rumplestiltskin arrives and magically perfects Jekyll's serum.
  Fact 0.53: Jekyll transforms into Hyde.
  Fact 0.54: Hyde blackmails Lydgate for Jekyll's membership.
  Fact 0.55: Mary is Lydgate's daughter and Jekyll's unrequited love.
  Fact 0.56: Mary and Hyde fall in love.
  Fact 0.57: Jekyll accidentally fatally defenestrates Mary.
  Fact 0.58: Jekyll frames Hyde for Mary's death.
  Fact 0.59: In Storybrooke, Gold imprisons Belle on the ship to protect her from Hyde.
  Fact 0.60: Jekyll tries to kill Belle in revenge for Rumplestiltskin's role in Mary's death.
  Fact 0.61: Hook saves Belle by killing Jekyll.
  Fact 0.62: Hyde dies of identical wounds.
  Fact 0.63: Regina asks Emma to kill her if necessary to stop the Evil Queen.
  Fact 0.64: Gold warns Belle that their child will need his protection.
  Fact 0.65: Mary Margaret resumes teaching.
  Fact 0.66: Mary Margaret's assistant Shirin inspires her to draw on her life as Snow.
  Fact 0.67: Shirin is secretly Princess Jasmine.
  Fact 0.68: Princess Jasmine is working with the Oracle to search for Aladdin.
  Fact 0.69: Episode 116, 'Street Rats', is season episode 5 and was directed by Norman Buckley and written by Edward Kitsis and Adam Horowitz.
  Fact 0.70: Episode 116 aired on October 23, 2016 and had 3.40 million U.S. viewers.
  Fact 0.71: In the past in Agrabah, Aladdin seeks a magical weapon for Jasmine to save the kingdom from Jafar.
  Fact 0.72: Aladdin is the weapon and a Savior.
  Fact 0.73: Jafar tempts Aladdin with magical shears that can sever him from his fate as the Savior and from his resulting death.
  Fact 0.74: Aladdin rejects Jafar and saves Jasmine.
  Fact 0.75: In Storybrooke, the Oracle is murdered.
  Fact 0.76: Emma helps Jasmine, hoping Aladdin's survival ensures her own.
  Fact 0.77: Aladdin has faked his death.
  Fact 0.78: Aladdin reveals himself to Emma.
  Fact 0.79: Aladdin confesses that he doomed Agrabah by using the shears.
  Fact 0.80: Aladdin gives the shears to Emma.
  Fact 0.81: Emma asks Hook to dispose of the shears.
  Fact 0.82: Hook secretly keeps the shears.
  Fact 0.83: The Evil Queen impersonates Archie to learn of Emma's visions.
  Fact 0.84: The Evil Queen convinces Zelena to turn Archie back into a cricket.
  Fact 0.85: Episode 117, 'Dark Waters', is season episode 6 and was directed by Robert Duncan and written by Andrew Chambliss and Brigitte Hales.
  Fact 0.86: Episode 117 aired on October 30, 2016 and had 3.06 million U.S. viewers.
  Fact 0.87: During the Dark Curse, Captain Nemo abducts Hook.
  Fact 0.88: Captain Nemo invites Hook to join the crew 'family' of his submarine, Nautilus.
  Fact 0.89: The crew is on a quest for the 'Mysterious Island', which is the Land of Untold Stories.
  Fact 0.90: Hook discovers the first mate is his half-brother, Liam.
  Fact 0.91: Hook decides to leave to avoid confrontation.
  Fact 0.92: Liam attacks Hook, instead mortally wounding Nemo, who steps in.
  Fact 0.93: In the present, Archie is freed.
  Fact 0.94: Henry confronts Hook over lying about the shears.
  Fact 0.95: Liam imprisons Henry and Hook on Nautilus.
  Fact 0.96: Hook opens up and Henry accepts him as family.
  Fact 0.97: Henry and Hook save each other and dispose of the shears.
  Fact 0.98: Modern medicine saves Nemo.
  Fact 0.99: Nemo, Liam and Hook make amends.
  Fact 0.100: Emma and Aladdin discuss being a Savior.
  Fact 0.101: Jasmine reveals that Agrabah vanished.
  Fact 0.102: Snow supports Belle at her ultrasound.
  Fact 0.103: The Evil Queen seduces Gold.
  Fact 0.104: The Evil Queen offers Gold the retrieved shears in exchange for help taking Snow's heart.
  Fact 0.105: Episode 118, 'Heartless', is season episode 7 and was directed by Ralph Hemecker and written by Jane Espenson.
  Fact 0.106: Episode 118 aired on November 6, 2016 and had 3.56 million U.S. viewers.
  Fact 0.107: In the past, fugitive Snow and shepherd David separately travel to Longbourn.
  Fact 0.108: Snow travels to Longbourn to flee the kingdom.
  Fact 0.109: David travels to Longbourn to sell his struggling farm.
  Fact 0.110: Without seeing each other's faces, David saves Snow from a bounty hunter.
  Fact 0.111: Snow repays David with money to save the farm.
  Fact 0.112: Snow decides not to flee.
  Fact 0.113: Their hands touch and a magical sapling is created.
  Fact 0.114: In the present, the Evil Queen threatens to destroy Storybrooke with River of Lost Souls water.
  Fact 0.115: The Evil Queen's threat is unless Mary Margaret and David hand over their hearts.
  Fact 0.116: Mary Margaret and David learn the sapling could stop the Queen.
  Fact 0.117: The Queen destroys the sapling just after they find it.
  Fact 0.118: The sapling showed Mary Margaret and David their previous meeting.
  Fact 0.119: Mary Margaret and David conclude that their love can endure anything.
  Fact 0.120: Mary Margaret and David surrender themselves.
  Fact 0.121: The Queen casts a sleeping curse on their hearts.
  Fact 0.122: Because they share Snow's heart, the curse affects both of them.
  Fact 0.123: Each can wake the other but immediately succumbs to the curse.
  Fact 0.124: They can never be awake together.
  Fact 0.125: Regina uses Gold's relationship with the Queen to separate the Queen from Zelena.
  Fact 0.126: Zelena earns Gold's enmity by exposing his schemes to Belle.
  Fact 0.127: Episode 119, 'I'll Be Your Mirror', is season episode 8 and was directed by Jennifer Lynch and written by Jerome Schwartz and Leah Fong.
  Fact 0.128: Episode 119 aired on November 13, 2016 and had 3.40 million U.S. viewers.
  Fact 0.129: Regina and Emma try to magically trap the Evil Queen in the World Behind the Mirror.
  Fact 0.130: The Evil Queen traps Regina and Emma in the World Behind the Mirror.
  Fact 0.131: The Dragon is already imprisoned in the World Behind the Mirror.
  Fact 0.132: Henry prepares for a date with Violet.
  Fact 0.133: The Queen impersonates Regina and tries to manipulate Henry.
  Fact 0.134: Henry recognizes the Queen.
  Fact 0.135: Emma and Regina discover Sidney was building a mirror portal out of the world.
  Fact 0.136: Emma and Regina try to finish the mirror portal.
  Fact 0.137: The Evil Queen forces the Dragon to attack Emma and Regina, using his heart.
  Fact 0.138: The Evil Queen tries to make Henry kill the Dragon with the Hammer of Hephaestus.
  Fact 0.139: Henry instead uses the Hammer of Hephaestus to free his mothers.
  Fact 0.140: Mary Margaret and David adjust to living in shifts.
  Fact 0.141: Belle asks Zelena to help her escape Storybrooke via magical portal.
  Fact 0.142: Belle and Zelena form an alliance with Aladdin and Jasmine.
  Fact 0.143: Aladdin steals the Genie's lamp and the Sorcerer's wand from Gold's shop.
  Fact 0.144: Gold takes the wand back from Zelena.
  Fact 0.145: Gold puts a magical tracking device on Belle.
  Fact 0.146: Gold asks the Queen to kill Zelena.
  Fact 0.147: The connection between Gold's and Zelena's hearts protects Zelena from Gold.
  Fact 0.148: Episode 120, 'Changelings', is season episode 9 and was directed by Mairzee Almas and written by David H. Goodman and Brian Ridings.
  Fact 0.149: Episode 120 aired on November 27, 2016 and had 3.28 million U.S. viewers.
  Fact 0.150: In the Enchanted Forest, Rumplestiltskin, while served by Belle, abducts a baby.
  Fact 0.151: Rumplestiltskin uses the abducted baby as bait for the Black Fairy, the mother who abandoned him.
  Fact 0.152: The Black Fairy escapes without answering Rumplestiltskin's questions.
  Fact 0.153: In the present, Belle receives advice from her unborn child.
  Fact 0.154: In the present, Gold becomes increasingly unhinged.
  Fact 0.155: The Evil Queen resents Gold's order to attack Zelena.
  Fact 0.156: The Evil Queen accelerates Belle's pregnancy.
  Fact 0.157: Mother Superior delivers Belle's son, Gideon.
  Fact 0.158: Mother Superior sends Gideon into hiding.
  Fact 0.159: Gold vows to find Gideon.
  Fact 0.160: The Genie has been freed.
  Fact 0.161: Aladdin takes the Genie's place.
  Fact 0.162: Zelena and Regina continue to feud.
  Fact 0.163: Emma obtains the sword from her vision.
  Fact 0.164: Episode 121, 'Wish You Were Here', is season episode 10 and was directed by Ron Underwood and written by Edward Kitsis and Adam Horowitz.
  Fact 0.165: Episode 121 aired on December 4, 2016 and had 3.27 million U.S. viewers.
  Fact 0.166: The Black Fairy abducts Gideon.
  Fact 0.167: Emma discovers the sword can kill the Evil Queen without hurting Regina.
  Fact 0.168: The Queen steals the lamp.
  Fact 0.169: The Queen wishes Emma to an alternate reality where Emma was never the Savior.
  Fact 0.170: Regina follows the Queen into the alternate reality.
  Fact 0.171: David and his allies take the lamp from the Queen.
  Fact 0.172: The Queen is turned into a serpent by the hooded figure.
  Fact 0.173: The hooded figure is revealed to be the adult Gideon.
  Fact 0.174: Jasmine wishes herself and Aladdin to Agrabah.
  Fact 0.175: In the 'wish realm', Emma lives as a princess.
  Fact 0.176: Regina traumatically restores Emma's memory in the wish realm.
  Fact 0.177: Regina frees Rumplestiltskin in exchange for a magic bean.
  Fact 0.178: The portal to Storybrooke closes while Regina is distracted by the appearance of Robin Hood.
  Fact 0.179: Episode 122, 'Tougher Than the Rest', is season episode 11 and was directed by Billy Gierhart and written by Edward Kitsis and Adam Horowitz.
  Fact 0.180: Episode 122 aired on March 5, 2017 and had 3.03 million U.S. viewers.
  Fact 0.181: In 1990 Minnesota, a young Emma is told by a teenager that she has the power to change her fate.
  Fact 0.182: In the present day, Emma and Regina miss their way out of the Wish realm.
  Fact 0.183: Emma and Regina find another option for escape when they come across August.
  Fact 0.184: August agrees to help create a new portal.
  Fact 0.185: Regina goes to find Robin Hood to see if he would have been better off without her.
  Fact 0.186: Regina and Robin are caught by the Sheriff of Nottingham.
  Fact 0.187: Regina and Robin are later caught by the Wish realm Rumple.
  Fact 0.188: Robin believes Regina when she tells him about their being in an alternate reality.
  Fact 0.189: Emma discovers that August was the teenager she met in Minnesota.
  Fact 0.190: Emma inspires August to complete the portal.
  Fact 0.191: Emma, Regina and Robin return to Storybrooke.
  Fact 0.192: Gideon explains to Gold and Belle his intention to kill Emma.
  Fact 0.193: Gideon intends to kill Emma so that he can acquire her savior powers and kill the Black Fairy.
  Fact 0.194: Gideon's intention has David furious and Belle concerned.
  Fact 0.195: Emma and Gideon finally meet for the battle.
  Fact 0.196: Emma survives, choosing her own fate.
  Fact 0.197: Gideon retreats, proclaiming this far from over.
  Fact 0.198: Gold and Belle decide to work together for the good of their son.
  Fact 0.199: Episode 123, 'Murder Most Foul', is season episode 12 and was directed by Morgan Beggs and written by Jerome Schwartz and Jane Espenson.
  Fact 0.200: Episode 123 aired on March 12, 2017 and had 3.06 million U.S. viewers.
  Fact 0.201: A flashback details how Robert, the father of David and James, was murdered.
  Fact 0.202: In the flashback, Robert learns what happened to James after James ran away from King George.
  Fact 0.203: With help from Rumple, Robert tracks James to Pleasure Island.
  Fact 0.204: King George cuts short Robert and James' reunion.
  Fact 0.205: King George orders Robert's execution.
  Fact 0.206: Robert's death affects David in the present day.
  Fact 0.207: David sees visions of Robert and asks Hook to help him, despite Hook being reformed.
  Fact 0.208: Hook becomes concerned that David is taking the investigation too far by going after Albert.
  Fact 0.209: Hook intervenes and brings David back to sanity.
  Fact 0.210: Hook is worried about whether David will give him his blessing to marry Emma.
  Fact 0.211: David gives Hook his blessing to marry Emma.
  Fact 0.212: Regina tries to get the Wish Realm Robin assimilated into Storybrooke.
  Fact 0.213: Regina questions why Robin agreed to follow her.
  Fact 0.214: Robin had something else planned, and he steals a chest from the vault.
  Fact 0.215: The truth about who murdered Robert is revealed when August gives Hook missing pages he took from Henry's book.
  Fact 0.216: Hook sees an illustration of David's dad and recognizes him.
  Fact 0.217: Hook recalls murdering David's dad during a robbery in the Enchanted Forest so that there was no witness to his crime.
  Fact 0.218: Episode 124, 'Ill-Boding Patterns', is season episode 13 and was directed by Ron Underwood and written by Andrew Chambliss and Dana Horgan.
  Fact 0.219: Episode 124 aired on March 19, 2017 and had 2.71 million U.S. viewers.
  Fact 0.220: During the First Ogres War, Beowulf leads a fearsome charge on the battlefield in the Enchanted Forest of old.
  Fact 0.221: After every last soldier had been slain, Beowulf surrenders to death just as Rumplestiltskin intervenes.
  Fact 0.222: Rumplestiltskin kills all the ogres.
  Fact 0.223: Beowulf is irked by Rumple's new title as a hero in the village.
  Fact 0.224: Beowulf creates a ruse to lead Rumple to his demise.
  Fact 0.225: Fearing he would not be able to protect the villagers without Dark Magic, Rumple bestows his dagger upon Baelfire as an insurance policy.
  Fact 0.226: Beowulf retreats after a turbulent exchange and a threat to frame Rumplestiltskin as a murderer.
  Fact 0.227: Feeling the pressure of Beowulf's threat, Bae commands his father to kill him.
  Fact 0.228: Rumple later doses Bae with a memory potion to erase the ordeal from his mind and restore his innocence.
  Fact 0.229: In Storybrooke, Rumple recognizes a familiar pattern in Gideon.
  Fact 0.230: Gideon makes strides in his quest to kill Emma Swan at the cost of the Blue Fairy's vitality.
  Fact 0.231: Hook worries about the repercussions of telling Emma his newly unveiled secret.
  Fact 0.232: Emma unknowingly forces a proposal after happening upon Hook's engagement ring.
  Fact 0.233: Robin of Locksley attempts to flee the town, but to no avail.
  Fact 0.234: The Evil Queen escapes her reptilian confinement.
  Fact 0.235: The Evil Queen indoctrinates Robin.
  Fact 0.236: Episode 125, 'Page 23', is season episode 14 and was directed by Kate Woods and written by David H. Goodman and Brigitte Hales.
  Fact 0.237: Episode 125 aired on March 26, 2017 and had 2.85 million U.S. viewers.
  Fact 0.238: In the Enchanted Forest, the Evil Queen is in search of a spell that will help her end Snow White for good.
  Fact 0.239: The Evil Queen's father has love in mind for his daughter and leads her on a quest to Cupid's Arrow.
  Fact 0.240: The Evil Queen rebrands the arrow to follow hatred.
  Fact 0.241: The Evil Queen discovers that the person she despises the most is herself.
  Fact 0.242: In Storybrooke, the Evil Queen and Regina face off in a final confrontation.
  Fact 0.243: When Regina wins the upper hand, she remembers the Evil Queen's self-loathing.
  Fact 0.244: Regina evenly splits the light and the darkness in their hearts, to share their love and hate.
  Fact 0.245: Regina's split allows the Evil Queen to reform.
  Fact 0.246: After Emma calls off the engagement, Hook decides to board the Nautilus with Captain Nemo.
  Fact 0.247: Hook later changes his mind following a conversation with Snow.
  Fact 0.248: Before Hook can return, Gideon hijacks the Nautilus.
  Fact 0.249: Gideon sends the Nautilus out of Storybrooke.
  Fact 0.250: Episode 126, 'A Wondrous Place', is season episode 15 and was directed by Steve Pearlman and written by Jane Espenson and Jerome Schwartz.
  Fact 0.251: Episode 126 aired on April 2, 2017 and had 2.80 million U.S. viewers.
  Fact 0.252: At a loss for hope, Regina suggests a 'girls' night out' for Emma with her and Snow.
  Fact 0.253: At the bar, Regina confides in the mixologist and sheds a tear.
  Fact 0.254: In the Enchanted Forest, the Nautilus is transported.
  Fact 0.255: Aladdin and Jasmine board the Nautilus.
  Fact 0.256: Aladdin and Jasmine have yet to locate Agrabah and are growing weary.
  Fact 0.257: After Hook and Jasmine confide in one another, the vessel is irrevocably damaged.
  Fact 0.258: Jasmine uses a genie wish to relocate the group to the island where Jafar is located.
  Fact 0.259: The group discovers a cabana belonging to Ariel and Prince Eric.
  Fact 0.260: Ariel has collected a genie lamp that houses Jafar.
  Fact 0.261: Jasmine sways Jafar into divulging the location of the hidden Agrabah.
  Fact 0.262: Jasmine defeats Jafar with magic dust.
  Fact 0.263: Through true love's kiss, Aladdin and Jasmine, along with Hook and Ariel, are transported to the restored city.
  Fact 0.264: Aladdin's genie curse is broken.
  Fact 0.265: Ariel supplies Hook with a means to reach Emma.
  Fact 0.266: Emma receives the message.
  Fact 0.267: Gideon, no longer masquerading as the mixologist, threatens to use Emma's tears to keep Hook from returning to Storybrooke.
  Fact 0.268: Gideon's threat is unless Emma helps him kill the Black Fairy.
  Fact 0.269: Flashbacks to Ariel and Jasmine's first meeting show the princess faltering under Jafar's trickery.
  Fact 0.270: Episode 127, 'Mother's Little Helper', is season episode 16 and was directed by Billy Gierhart, with a story by Edward Kitsis and Adam Horowitz and a teleplay by Paul Karp.
  Fact 0.271: Episode 127 aired on April 9, 2017 and had 2.60 million U.S. viewers.
  Fact 0.272: In Storybrooke, Gold reveals to Emma and Snow that the Black Fairy is his mother.
  Fact 0.273: Gold reveals to Emma and Snow that the Dark Curse Regina cast was created by the Black Fairy.
  Fact 0.274: Emma agrees to help Gideon kill the Black Fairy.
  Fact 0.275: Gideon double-crosses Emma.
  Fact 0.276: Gideon leaves Emma to die at the hands of a giant spider.
  Fact 0.277: Gideon opens a portal with the sword.
  Fact 0.278: The portal allows the Black Fairy to slip through into Storybrooke.
  Fact 0.279: In the Enchanted Forest, Hook wagers the Jolly Roger for a magic bean in a game of cards with Blackbeard.
  Fact 0.280: As Blackbeard and Hook go through the portal, it malfunctions thanks to Gideon.
  Fact 0.281: Blackbeard and Hook end up in Neverland.
  Fact 0.282: Blackbeard and Hook are on the run from the remnants of Peter Pan's Lost Boys.
  Fact 0.283: In Storybrooke, Henry experiences a disorienting shift in his Author powers.
  Fact 0.284: Henry and Regina seek help from the previous Author, Isaac.
  Fact 0.285: Isaac reveals to Regina and Henry that they are nearing the end of the story.
  Fact 0.286: In the story, the savior must fight the Final Battle.
  Fact 0.287: Flashbacks reveal the abuses the Black Fairy inflicted on Gideon.
  Fact 0.288: The Black Fairy ripped out Gideon's heart.
  Fact 0.289: The Black Fairy is controlling Gideon to kill Emma.
  Fact 0.290: Episode 128, 'Awake', is season episode 17 and was directed by Sharat Raju and written by Andrew Chambliss and Leah Fong.
  Fact 0.291: Episode 128 aired on April 16, 2017 and had 2.51 million U.S. viewers.
  Fact 0.292: Regina works feverishly to wake Snow and Charming from the Sleeping Curse.
  Fact 0.293: Gold comes face to face with the Black Fairy.
  Fact 0.294: The Black Fairy is the powerful enemy Emma must fight in the Final Battle.
  Fact 0.295: In Neverland, Hook meets up with Tiger Lily.
  Fact 0.296: Tiger Lily gives Hook a piece of an ancient fairy wand to aid Emma in defeating the Black Fairy.
  Fact 0.297: Gold confronts the Black Fairy over her manipulating Gideon with his heart.
  Fact 0.298: The Black Fairy maintains that Storybrooke would be destroyed if she and Gold fought.
  Fact 0.299: A pixie flower allows Emma to open a doorway to Neverland.
  Fact 0.300: Emma rescues Hook.
  Fact 0.301: Hook re-proposes properly.
  Fact 0.302: The townsfolk gather around Emma's sleeping parents.
  Fact 0.303: The townsfolk each take small amounts of the Sleeping Curse unto themselves, diluting it.
  Fact 0.304: Diluting the curse allows Snow and David to finally reunite.
  Fact 0.305: In 1993, a pixie flower awakens Snow and Charming from their cursed selves.
  Fact 0.306: An awakened Rumplestiltskin tells Snow and Charming they must make a difficult decision to ensure that Emma fulfills her destiny as the Savior.
  Fact 0.307: Episode 129, 'Where Bluebirds Fly', is season episode 18 and was directed by Michael Schultz and written by David H. Goodman and Brigitte Hales.
  Fact 0.308: Episode 129 aired on April 23, 2017 and had 2.69 million U.S. viewers.
  Fact 0.309: In Zelena's past, her childhood friend Stanum asks for her help in finding the Crimson Heart.
  Fact 0.310: The Crimson Heart is a magical artifact that would restore Stanum's human form.
  Fact 0.311: Stanum was cursed by the Wicked Witch of the North to slowly turn into tin.
  Fact 0.312: Zelena refuses to give up her magic.
  Fact 0.313: Zelena betrays Stanum by keeping the Heart.
  Fact 0.314: Zelena leaves Stanum to become the Tin Man.
  Fact 0.315: In the present, the Black Fairy tempts Zelena with an offer to join her.
  Fact 0.316: Zelena opposes the Black Fairy instead.
  Fact 0.317: Regina is brought into the chaos by trying to stop her sister.
  Fact 0.318: Regina and Zelena find the Black Fairy and Gideon in the crystal mines.
  Fact 0.319: The Black Fairy tricks Zelena into utilizing her unstable Dark Magic to strengthen her for the Final Battle.
  Fact 0.320: Zelena realizes there is nothing left for her back in Oz.
  Fact 0.321: Zelena decides to help the residents fight back against the Black Fairy.
  Fact 0.322: Zelena uses the Crimson Heart to give up her magic.
  Fact 0.323: Giving up her magic undoes the mess Zelena made in the mines.
  Fact 0.324: Emma uses her magic and one of the crystals to bring Mother Superior out of her coma.
  Fact 0.325: Regina and Zelena help Emma bring Mother Superior out of her coma.
  Fact 0.326: Episode 130, 'The Black Fairy', is season episode 19 and was directed by Alrick Riley and written by Jerome Schwartz and Dana Horgan.
  Fact 0.327: Episode 130 aired on April 30, 2017 and had 3.05 million U.S. viewers.
  Fact 0.328: The origin of Rumplestiltskin's true destiny is revealed.
  Fact 0.329: Rumplestiltskin's mother, Fiona, learns from the Blue Fairy and Tiger Lily that he was destined to become The Savior.
  Fact 0.330: Fiona fears for her son and chooses power over protecting Rumple.
  Fact 0.331: Fiona becomes the Black Fairy.
  Fact 0.332: The Blue Fairy banishes Fiona to the Dark Realm.
  Fact 0.333: In the present, everyone looks for the missing wand.
  Fact 0.334: Gold brings Gideon and Emma into the Dream World to search for Gideon's heart.
  Fact 0.335: The events of the flashbacks unfold in the Dream World.
  Fact 0.336: Gold confronts Fiona.
  Fact 0.337: Fiona reveals the real reason behind why Gold was abandoned.
  Fact 0.338: Gold retrieves Gideon's heart.
  Fact 0.339: Gold tells everyone that the Black Fairy is dead.
  Fact 0.340: The Black Fairy's death paves the way for Emma and Hook's wedding.
  Fact 0.341: Gold and Fiona make a deal that will see the Final Battle go forward.
  Fact 0.342: Emma will be the Savior that fights in the Final Battle.
  Fact 0.343: Regina helps Zelena adjust to life without her powers.
  Fact 0.344: Episode 131, 'The Song in Your Heart', is season episode 20 and was directed by Ron Underwood and written by David H. Goodman and Andrew Chambliss.
  Fact 0.345: Episode 131 aired on May 7, 2017 and had 2.87 million U.S. viewers.
  Fact 0.346: In the past, Snow makes a wish that grants the entire Enchanted Forest the ability to use the gift of song as a strength.
  Fact 0.347: The wish also spreads to Oz.
  Fact 0.348: Regina becomes affected by the wish.
  Fact 0.349: Regina finds a way to end the wish before the first curse takes place.
  Fact 0.350: The Blue Fairy, who granted Snow the wish, tells Snow and David that the wish will be part of Emma's destiny.
  Fact 0.351: In Emma's past, her chance to make use of her talent is held back by being told she'll be alone.
  Fact 0.352: The weakness of being told she'll be alone comes back to taunt Emma in the present day.
  Fact 0.353: Fiona returns to use that weakness to take Emma's heart.
  Fact 0.354: Fiona unleashes a new curse on Emma's wedding day, with help from Gold.
  Fact 0.355: The curse freezes Emma's family.
  Fact 0.356: Henry discovers the page in the book that reveals the song in Emma's heart will make her stronger.
  Fact 0.357: The song embolden Emma.
  Fact 0.358: The song stops Fiona from crushing Emma's heart.
  Fact 0.359: Fiona tells Emma that the battle will continue.
  Fact 0.360: On her wedding day, Emma and Hook finally marry.
  Fact 0.361: Dark clouds of the Black Fairy's curse erupt from the clock tower at 6:00 PM.
  Fact 0.362: The dark clouds engulf Emma and Hook.
  Fact 0.363: The curse sets up the final battle.
  Fact 0.364: Season episodes 21 and 22, 'The Final Battle', were directed by Steve Pearlman and Ralph Hemecker and written by Edward Kitsis and Adam Horowitz.
  Fact 0.365: The Final Battle aired on May 14, 2017 and had 2.95 million U.S. viewers.
  Fact 0.366: Fiona's curse alters Storybrooke.
  Fact 0.367: Fiona's curse institutionalizes Emma.
  Fact 0.368: Fiona's curse leaves only Henry and Gold 'awake'.
  Fact 0.369: Fiona's curse sends Emma's other allies to their home realms.
  Fact 0.370: The Final Battle begins over Emma's belief.
  Fact 0.371: If Emma stops believing in the realms of story, they will be consumed by the Darkness and cease to exist.
  Fact 0.372: If all the realms are destroyed, Fiona would gain unchecked powers.
  Fact 0.373: If all the realms are destroyed, Fiona would be able to break the Laws of Magic at will.
  Fact 0.374: In the Enchanted Forest, the banished characters gather at Regina's castle.
  Fact 0.375: David and Hook obtain a magic bean.
  Fact 0.376: The Evil Queen sacrifices herself.
  Fact 0.377: The Evil Queen holds off the Darkness to delay the group's destruction.
  Fact 0.378: Fiona convinces Emma to destroy the book.
  Fact 0.379: Henry hand-draws a new book.
  Fact 0.380: Henry's hand-drawn book restores Emma's belief in time to save everyone in the castle.
  Fact 0.381: Gold kills Fiona.
  Fact 0.382: Fiona had hidden Belle and turned her into a coward.
  Fact 0.383: Gold's killing of Fiona breaks the Curse.
  Fact 0.384: Breaking the Curse restores everyone's memories.
  Fact 0.385: Breaking the Curse reunites everyone in Storybrooke.
  Fact 0.386: Gideon is still under orders to kill Emma.
  Fact 0.387: Gold attempts to free Gideon's heart.
  Fact 0.388: A spell cast by the Black Fairy prevents Gold from freeing Gideon's heart.
  Fact 0.389: Emma lays down her sword, allowing Gideon to impale her.
  Fact 0.390: Gideon impaling Emma releases Emma's light magic and ends the battle.
  Fact 0.391: Gold and Emma have both chosen good.
  Fact 0.392: Gideon is returned to infancy.
  Fact 0.393: The realms are restored.
  Fact 0.394: Those who died in the crisis are revived.
  Fact 0.395: The characters embark on their 'happy beginnings'.
  Fact 0.396: Many years later in the New Enchanted Forest, an adult Henry vanishes while defending his daughter Lucy from an unidentified entity.
  Fact 0.397: Guided by Tiger Lily, Lucy travels to Seattle.
  Fact 0.398: In Seattle, Henry no longer remembers Lucy.

Retrieved items:
<item 1>
Once Upon a Time (season 6)
the Bow '' `` Nimue '' `` Birth '' `` The Bear King '' `` Broken Heart '' `` Swan Song '' `` Souls of the Departed '' `` Labor of Love '' `` Devil 's Due '' ``
The Brothers Jones '' `` Our Decay '' `` Her Handsome Hero '' `` Ruby Slippers '' `` Sisters '' `` Firebird '' `` Last Rites '' `` Only You '' `` An Untold
Story '' Season 6 `` The Savior '' `` A Bitter Draught '' `` The Other Shoe '' `` Strange Case '' `` Street Rats '' `` Dark Waters '' `` Heartless '' `` I 'll
Be Your Mirror '' `` Changelings '' `` Wish You Were Here '' `` Tougher Than the Rest '' `` Murder Most Foul '' `` Ill - Boding Patterns '' `` Page 23 '' `` A
Wondrous Place '' `` Mother 's Little Helper '' `` Awake '' `` Where Bluebirds Fly '' `` The Black Fairy '' `` The Song in Your Heart '' `` The Final Battle ''
Season 7 `` Hyperion Heights '' `` A Pirate 's Life '' `` The Garden of Forking Paths '' `` Beauty '' `` Greenbacks '' `` Wake Up Call '' `` Eloise Gardener ''
`` Pretty
</item 1>

<item 2>
Once Upon a Time (season 6)
Gold kills Fiona , who had hidden Belle and turned her into a coward ; this breaks the Curse , which restores everyone 's memories and reunites them in
Storybrooke . However , Gideon is still under orders to kill Emma . Gold attempts to free Gideon 's heart ; however , a spell cast by the Black Fairy prevents
him from doing so . Emma lays down her sword , allowing Gideon to impale her ; this releases Emma 's light magic and ends the battle . With both Gold and Emma
having chosen good , Gideon is returned to infancy and the realms are restored , and those who died in the crisis are revived . The characters embark on their
`` happy beginnings . '' Many years later in the New Enchanted Forest , an adult Henry vanishes while defending his daughter Lucy from an unidentified entity .
Guided by Tiger Lily , Lucy travels to Seattle , where Henry no longer remembers her .
</item 2>

<item 3>
Once Upon a Time (season 6)
Emma agrees to help Gideon kill the Black Fairy , but is double - crossed by Gideon and left to die at the hands of a giant spider . Gideon opens a portal with
the sword , allowing the Black Fairy to slip through into Storybrooke . Meanwhile in the Enchanted Forest , Hook wagers the Jolly Roger for a magic bean in a
game of cards with Blackbeard . As Blackbeard and Hook go through the portal , it malfunctions thanks to Gideon and the duo ends up in Neverland , on the run
from the remnants of Peter Pan 's Lost Boys . Back in Storybrooke , Henry experiences a disorienting shift in his Author powers , so he and Regina seek help
from the only person who can aid them : the previous Author , Isaac . Isaac reveals to Regina and Henry that they are nearing the end of the story in which the
savior must fight the Final Battle . Flashbacks reveal the abuses the Black Fairy inflicted on Gideon and that the Black Fairy ripped out Gideon 's heart and is
controlling him to kill Emma .
</item 3>

<item 4>
Once Upon a Time (season 6)
Regina is brought into the chaos by trying to stop her sister and they find the Black Fairy and Gideon in the crystal mines , with the Black Fairy tricking
Zelena into utilizing her unstable Dark Magic to strengthen her for the Final Battle . Realizing that there is nothing left for her back in Oz , Zelena decides
to help the residents fight back against the Black Fairy by using the Crimson Heart to give up her magic , thereby undoing the mess she made in the mines . Emma
then uses her magic and one of the crystals to bring Mother Superior out of her coma with some help from Regina and Zelena . 130 19 `` The Black Fairy '' Alrick
Riley Jerome Schwartz & Dana Horgan April 30 , 2017 ( 2017 - 04 - 30 ) 3.05 The origin of Rumplestiltskin 's true destiny is revealed , as his mother Fiona
learns from the Blue Fairy and Tiger Lily that he was destined to become The Savior . When Fiona fears for her son and chooses power over protecting Rumple , it
results in her becoming the Black Fairy , and the Blue Fairy banishing her to the Dark Realm .
</item 4>

<item 5>
Once Upon a Time (season 6)
overall No. in season Title Directed by Written by Original air date U.S. viewers ( millions ) 112 `` The Savior '' Eagle Egilsson Edward Kitsis & Adam Horowitz
September 25 , 2016 ( 2016 - 09 - 25 ) 3.99 In the past , Aladdin , a `` Savior '' , is taunted by Jafar over the inevitable downfall of all Saviors . In
Storybrooke , refugees arrive from the Land of Untold Stories . Emma experiences tremors and visions ; an arrested Hyde helps her receive an oracle 's vision of
a hooded figure killing her . Guided by Morpheus , Gold enters Belle 's dreams to lift the sleeping curse ; he tries to renew their love , but she rejects him .
`` Morpheus '' reveals he is really their unborn son , and desired this outcome . Belle wakes and leaves Gold . Henry convinces Regina that Hades may have lied
about Robin Hood 's soul having been destroyed .
</item 5>

<item 6>
Once Upon a Time (season 6)
Once Upon a Time ( season 6 ) - wikipedia Once Upon a Time ( season 6 ) Jump to : navigation , search Once Upon a Time ( season 6 ) DVD cover Starring Ginnifer
Goodwin Jennifer Morrison Lana Parrilla Josh Dallas Emilie de Ravin Colin O'Donoghue Jared S. Gilmore Rebecca Mader Robert Carlyle Country of origin United
States No. of episodes 22 Release Original network ABC Original release September 25 , 2016 ( 2016 - 09 - 25 ) -- May 14 , 2017 ( 2017 - 05 - 14 ) Season
chronology ← Previous Season 5 Next → Season 7 List of Once Upon a Time episodes The sixth season of the American ABC fantasy - drama Once Upon a Time was
ordered on March 3 , 2016 . It debuted on September 25 , 2016 , and concluded on May 14 , 2017 . In January 2017 , it was stated that the sixth season would end
the main storyline , and for a seventh season , the series would be softly rebooted with a new storyline .
</item 6>

<item 7>
Once Upon a Time (season 6)
Contents ( hide ) 1 Premise 2 Cast and characters 2.1 Regular 2.2 Recurring 2.3 Guest 3 Episodes 4 Production 4.1 Development 4.1. 1 Musical episode 4.2 Casting
5 Ratings 6 References 7 External links Premise ( edit ) The residents of Storybrooke are faced with threats from the Evil Queen and Mr. Hyde , both with
different agendas following the arrival of refugees from the Land of Untold Stories . This leads to events to test Emma 's savior abilities that results in the
arrival of the Black Fairy to begin the Final Battle that was prophesied centuries ago . As dark and light collides , a new , and darker curse is unleashed ,
leading to the fall of all the realms with magic . The Final Battle is fought and won , leading to the heroes earning their happy beginnings , but a new
adventure begins in the far future for an adult Henry involving his daughter Lucy , and another curse looming across the New Enchanted Forest inhabitants .
</item 7>

<item 8>
Once Upon a Time (season 6)
132 133 21 22 `` The Final Battle '' Steve Pearlman Ralph Hemecker Edward Kitsis & Adam Horowitz May 14 , 2017 ( 2017 - 05 - 14 ) 2.95 Fiona 's curse alters
Storybrooke , institutionalizing Emma and leaving only Henry and Gold `` awake , '' while sending Emma 's other allies to their home realms . The Final Battle
begins over Emma 's belief ; if she stops believing in the realms of story , they will be consumed by the Darkness and cease to exist . If all the realms are
destroyed , Fiona would gain unchecked powers and be able to break the Laws of Magic at will . In the Enchanted Forest , the banished characters gather at
Regina 's castle ; David and Hook obtain a magic bean and the Evil Queen sacrifices herself , holding off the Darkness to delay the group 's destruction . Fiona
convinces Emma to destroy the book , but Henry hand - draws one that restores her belief in time to save everyone in the castle .
</item 8>

<item 9>
Once Upon a Time (season 6)
-- Executive producers Kitsis and Horowitz on Once 's interpretation of Aladdin . Emilie de Ravin told fans that she will be returning for the sixth season . It
was also announced later that Sam Witwer and Hank Harris have signed a contract to play two new recurring characters on the sixth season of the show . On March
29 , 2016 , Lana Parrilla confirmed she would be returning for the sixth season . Robert Carlyle was confirmed to be returning for the sixth season as
Rumplestiltskin along with Rebecca Mader as Zelena and Jared Gilmore as Henry Mills . It was announced that Giles Matthey was cast as Morpheus , who is slated
to appear in the first episode of the season . On July 20 , it was announced that Craig Horner would be portraying the Count of Monte Cristo , who was
introduced in the second episode of the season .
</item 9>

<item 10>
Once Upon a Time (season 6)
Existing fictional characters introduced to the series during the season include Aladdin , Princess Jasmine , the Count of Monte Cristo , Captain Nemo , Lady
Tremaine , Beowulf , Tiger Lily and the Tin Man . Original new characters include Gideon , the Black Fairy , Mary Lydgate , and Robert . The show also
reintroduced Jafar and Dr. Arthur Lydgate , who previously appeared in Once Upon a Time in Wonderland . This season also marks the final appearance of Emma Swan
( Jennifer Morrison ) as a series regular . Morrison announced she would be departing the series after the sixth - season finale , but if the series receives a
seventh season renewal she has agreed to appear in at least one episode . After serving as a series regular for two seasons , Rebecca Mader also announced that
season six would be her last on the show as a regular . Shortly after , Adam Horowitz and Edward Kitsis announced that original cast members Ginnifer Goodwin ,
Josh Dallas , and Jared Gilmore , as well as Emilie de Ravin who joined the main cast in season two , would also exit the show at the end of the season .
</item 10>
=== END ITEM id=e0b5eae77f49be09 ===

=== ITEM id=206cf87f09f7df29 ===
Gold sentences and facts:
Sentence 0: The Sino - Indian War ( Hindi : भारत - चीन युद्ध Bhārat - Chīn Yuddh ) , also known as the Sino - Indian Border Conflict ( simplified Chinese : 中 印 边境 战争 ; traditional Chinese : 中 印 邊境 戰爭 ; pinyin : Zhōng - Yìn Biānjìng Zhànzhēng ) , was a war between China and India that occurred in 1962 . A disputed Himalayan border was the main pretext for war , but other issues played a role . There had been a series of violent border incidents after the 1959 Tibetan uprising , when India had granted asylum to the Dalai Lama . India initiated a Forward Policy in which it placed outposts along the border , including several north of the McMahon Line , the eastern portion of a Line of Actual Control proclaimed by Chinese Premier Zhou Enlai in 1959 .
  Fact 0.0: The Sino-Indian War is also known as the Sino-Indian Border Conflict.
  Fact 0.1: The Sino-Indian War's Hindi name is भारत - चीन युद्ध (Bhārat - Chīn Yuddh).
  Fact 0.2: The Sino-Indian War's simplified Chinese name is 中印边境战争.
  Fact 0.3: The Sino-Indian War's traditional Chinese name is 中印邊境戰爭.
  Fact 0.4: The Sino-Indian War's pinyin name is Zhōng-Yìn Biānjìng Zhànzhēng.
  Fact 0.5: The Sino-Indian War was a war between China and India.
  Fact 0.6: The Sino-Indian War occurred in 1962.
  Fact 0.7: A disputed Himalayan border was the main pretext for the Sino-Indian War.
  Fact 0.8: Other issues played a role in the Sino-Indian War.
  Fact 0.9: There had been a series of violent border incidents after the 1959 Tibetan uprising.
  Fact 0.10: India had granted asylum to the Dalai Lama at the time of the 1959 Tibetan uprising.
  Fact 0.11: India initiated a Forward Policy.
  Fact 0.12: In its Forward Policy, India placed outposts along the border.
  Fact 0.13: In its Forward Policy, India placed several outposts north of the McMahon Line.
  Fact 0.14: The McMahon Line is the eastern portion of a Line of Actual Control.
  Fact 0.15: The Line of Actual Control was proclaimed by Chinese Premier Zhou Enlai in 1959.

Retrieved items:
<item 1>
Sino-Indian War
India 's stance that China withdraw from Aksai Chin caused continual deterioration of the diplomatic situation to the point that internal forces were pressuring
Nehru to take a military stance against China . 1960 meetings to resolve the boundary question ( edit ) In 1960 , based on an agreement between Nehru and Zhou
Enlai , officials from India and China held discussions in order to settle the boundary dispute . China and India disagreed on the major watershed that defined
the boundary in the western sector . The Chinese statements with respect to their border claims often misrepresented the cited sources . The Forward policy (
edit ) At the beginning of 1961 , Nehru appointed General B.M. Kaul as army Chief of General Staff , but he refused to increase military spending and prepare
for a possible war . According to James Barnard Calvin of the U.S. Navy , in 1959 , India started sending Indian troops and border patrols into disputed areas .
This program created both skirmishes and deteriorating relations between India and China . The aim of this policy was to create outposts behind advancing
Chinese troops to interdict their supplies , forcing them north of the disputed line .
</item 1>

<item 2>
Sino-Indian War
Unable to reach political accommodation on disputed territory along the 3,225 - kilometre - long Himalayan border , the Chinese launched simultaneous offensives
in Ladakh and across the McMahon Line on 20 October 1962 , coinciding with the Cuban Missile Crisis . Chinese troops advanced over Indian forces in both
theatres , capturing Rezang la in Chushul in the western theatre , as well as Tawang in the eastern theatre . The war ended when China declared a ceasefire on
20 November 1962 , and simultaneously announced its withdrawal to its claimed ' line of actual control ' . The Sino - Indian War is notable for the harsh
mountain conditions under which much of the battle took place , entailing large - scale combat at altitudes of over 4,000 metres ( 14,000 feet ) . The Sino -
Indian War was also noted for the non-deployment of the navy or air force by either the Chinese or Indian side .
</item 2>

<item 3>
Sino-Indian War
It is noteworthy that the buildup and offensive from China occurred concurrently with the 13 - day Cuban Missile Crisis ( 16 -- 28 October 1962 ) that saw both
the United States and the Soviet Union confronting each other , and India did not receive assistance from either of these world powers until the Cuban Missile
Crisis was resolved . Contents ( hide ) 1 Location 2 Background 2.1 Aksai Chin 2.2 The McMahon Line 3 Events leading up to war 3.1 Tibet and the border dispute
3.2 1960 meetings to resolve the boundary question 3.3 The Forward Policy 3.4 Early incidents 4 Confrontation at Thag La 5 Chinese and Indian preparations 5.1
Motives 5.2 Military planning 6 Chinese offensive 6.1 Eastern theatre 6.2 Western theatre 7 Lull in the fighting 8 Continuation of war 8.1 Eastern theatre 8.2
Western theatre 9 Ceasefire 10 World opinion 11 Involvement of other nations 12 Aftermath 12.1 China 12.2 India 12.2.
</item 3>

<item 4>
Sino-Indian War
Xu Yan , prominent Chinese military historian and professor at the PLA 's National Defense University , gives an account of the Chinese leadership 's decision
to go to war . By late September 1962 , the Chinese leadership had begun to reconsider their policy of `` armed coexistence '' , which had failed to address
their concerns with the forward policy and Tibet , and consider a large , decisive strike . On 22 September 1962 , the People 's Daily published an article
which claimed that `` the Chinese people were burning with ' great indignation ' over the Indian actions on the border and that New Delhi could not ' now say
that warning was not served in advance ' . '' Military planning ( edit ) The Indian side was confident war would not be triggered and made little preparations .
India had only two divisions of troops in the region of the conflict . In August 1962 , Brigadier D.K. Palit claimed that a war with China in the near future
could be ruled out . Even in September 1962 , when Indian troops were ordered to `` expel the Chinese '' from Thag La , Maj . General J.S.
</item 4>

<item 5>
Sino-Indian War
( March 2017 ) ( Learn how and when to remove this template message ) Sino - Indian War ( 1962 ) Remembering a War : The 1962 India - China Conflict --
Rediff.com . Neville Maxwell : Henderson Brooks Report 1962 Sino - Indian War , Hindustan Times Why India lost the 1962 border war ? -- Tejas Patel War in the
Himalayas : 1962 Indo - Sino Conflict ( includes official war history ) from History Division , Ministry of Defence , Government of India Critical Asian Studies
Article : Sino Indian War 1962 India , China to speed up border dispute talks : 2005 Xinhuanet The Rediff Special / Claude Arpi 1962 War and Its Implications fr
Sino - India Relations Historical maps of the Sino - Indian border ( in Chinese ) Conflict in Kashmir : Selected Internet Resources by the Library , University
of California , Berkeley , USA ; ( ( University of California , Berkeley Library Bibliographies and Web - Bibliographies list ) Frontier India India - China
Section China , India , and the fruits of Nehru 's folly by Venkatesan Vembu , Daily News & Analysis , 6 June
</item 5>

<item 6>
Sino-Indian War
The Sino - Indian War ( Hindi : भारत - चीन युद्ध Bhārat - Chīn Yuddh ) , also known as the Sino - Indian Border Conflict ( simplified Chinese : 中 印 边境 战争 ;
traditional Chinese : 中 印 邊境 戰爭 ; pinyin : Zhōng - Yìn Biānjìng Zhànzhēng ) , was a war between China and India that occurred in 1962 . A disputed Himalayan
border was the main pretext for war , but other issues played a role . There had been a series of violent border incidents after the 1959 Tibetan uprising ,
when India had granted asylum to the Dalai Lama . India initiated a Forward Policy in which it placed outposts along the border , including several north of the
McMahon Line , the eastern portion of a Line of Actual Control proclaimed by Chinese Premier Zhou Enlai in 1959 .
</item 6>

<item 7>
Sino-Indian War
According to Patterson , the Indian motives were threefold : Test Chinese resolve and intentions regarding India . Test whether India would enjoy Soviet backing
in the event of a Sino - Indian war . Create sympathy for India within the U.S. , with whom relations had deteriorated after the Indian annexation of Goa . On
10 July 1962 , 350 Chinese troops surrounded an Indian post in Chushul ( north of the McMahon Line ) but withdrew after a heated argument via loudspeaker . On
22 July , the Forward Policy was extended to allow Indian troops to push back Chinese troops already established in disputed territory . Whereas Indian troops
were previously ordered to fire only in self - defence , all post commanders were now given discretion to open fire upon Chinese forces if threatened . In
August , the Chinese military improved its combat readiness along the McMahon Line and began stockpiling ammunition , weapons and gasoline . Given his
foreknowledge of the coming Cuban Missile Crisis , Mao Zedong was able to persuade Nikita Khrushchev to reverse the Russian policy of backing India , at least
temporarily . In mid-October , the Communist organ Pravda encouraged peace between India and China .
</item 7>

<item 8>
Sino-Indian War
Sino - Indian war - Wikipedia Sino - Indian war Jump to : navigation , search The examples and perspective in this article may not include all significant
viewpoints . Please improve the article or discuss the issue . ( September 2017 ) ( Learn how and when to remove this template message ) Sino - Indian War The
Sino - Indian War was fought between India and China Date 20 October -- 21 November 1962 ( 1 month and 1 day ) Location Aksai Chin and North - East Frontier
Agency Result Chinese victory Belligerents India China Commanders and leaders Brij Mohan Kaul ( Chief of General Staff of the Indian Army ) Sarvepalli
Radhakrishnan ( President of India ) Jawaharlal Nehru ( Prime Minister of India ) V.K.
</item 8>

<item 9>
Sino-Indian War
China 's heavy artillery facing Taiwan could then be moved to Tibet . It took China six to eight months to gather the resources needed for the war , according
to Anil Athale , author of the official Indian history . The Chinese sent a large quantity of non-military supplies to Tibet through the Indian port of Calcutta
. Early incidents ( edit ) Various border conflicts and `` military incidents '' between India and China flared up throughout the summer and autumn of 1962 . In
May , the Indian Air Force was told not to plan for close air support , although it was assessed as being a feasible way to counter the unfavourable ratio of
Chinese to Indian troops . In June , a skirmish caused the deaths of dozens of Chinese troops . The Indian Intelligence Bureau received information about a
Chinese buildup along the border which could be a precursor to war . During June -- July 1962 , Indian military planners began advocating `` probing actions ''
against the Chinese , and accordingly , moved mountain troops forward to cut off Chinese supply lines . According to Patterson , the Indian motives were
threefold : Test Chinese resolve and intentions regarding India .
</item 9>

<item 10>
Sino-Indian War
Retrieved 4 November 2010 . Bibliography ( edit ) Das Gupta , Amit R. ; Lüthi , Lorenz M. , eds. ( 2016 ) , The Sino - Indian War of 1962 : New perspectives ,
Taylor & Francis , ISBN 978 - 1 - 315 - 38892 - 2 Lüthi , Lorenz M. ( 2016 ) , `` India 's Relations with China , 1945 -- 1974 '' , Ibid , pp. 29 -- 47 Chaowu ,
Dai ( 2016 ) , `` From ' Hindi - Chini Bhai - Bhai ' to ' international class struggle ' against Nehru : China 's India policy and the frontier dispute , 1950
-- 62 '' , Ibid , pp. 68 -- 84 Fisher , Margaret W. ; Rose , Leo E. ; Huttenback , Robert A.
</item 10>
=== END ITEM id=206cf87f09f7df29 ===

=== ITEM id=bcb242e830d13a9b ===
Gold sentences and facts:
Sentence 0: There are several major additional characters introduced by the television series and numerous narrative details and the plotline differ radically from the source novel . For example , the planned Nazi pre-emptive nuclear strike on Japan , `` Operation Dandelion , '' is apparently being prevented only by Hitler 's personal refusal to authorise it , leading Heydrich and the faction demanding pre-emptive war to plot the Führer 's assassination . In addition , Hawthorne Abendsen does not appear in the first season of the television version and The Grasshopper Lies Heavy is a series of newsreel films depicting multiple alternative realities rather than a novel ( although this idea may actually be borrowed from Dick 's later novel Valis which features a mysterious film depicting yet another dystopian alternative history of the USA ) . As of the Season 1 finale , these films are being tracked down by SS agents like Blake for dispatch to Hitler for an as - yet - unknown purpose .
  Fact 0.0: The television series introduces several major additional characters.
  Fact 0.1: The narrative details of the television series differ radically from the source novel.
  Fact 0.2: The plotline of the television series differs radically from the source novel.
  Fact 0.3: In the television series, the planned Nazi pre-emptive nuclear strike on Japan, 'Operation Dandelion', is apparently being prevented only by Hitler's personal refusal to authorise it.
  Fact 0.4: Hitler's refusal to authorise Operation Dandelion leads Heydrich and the faction demanding pre-emptive war to plot the Führer's assassination.
  Fact 0.5: Hawthorne Abendsen does not appear in the first season of the television version.
  Fact 0.6: In the television version, The Grasshopper Lies Heavy is a series of newsreel films depicting multiple alternative realities.
  Fact 0.7: In the television version, The Grasshopper Lies Heavy is not a novel.
  Fact 0.8: The idea of The Grasshopper Lies Heavy as newsreel films may actually be borrowed from Dick's later novel Valis.
  Fact 0.9: The novel Valis features a mysterious film depicting yet another dystopian alternative history of the USA.
  Fact 0.10: As of the Season 1 finale, these films are being tracked down by SS agents like Blake.
  Fact 0.11: The films are being tracked down for dispatch to Hitler.
  Fact 0.12: The films are being dispatched to Hitler for an as-yet-unknown purpose.

Retrieved items:
<item 1>
The Man in the High Castle
In addition , Hawthorne Abendsen does not appear in the first season of the television version and The Grasshopper Lies Heavy is a series of newsreel films
depicting multiple alternative realities rather than a novel ( although this idea may actually be borrowed from Dick 's later novel Valis which features a
mysterious film depicting yet another dystopian alternative history of the USA ) . As of the Season 1 finale , these films are being tracked down by SS agents
like Blake for dispatch to Hitler for an as - yet - unknown purpose . In season two we see a map on John Smith 's wall of the world . From this map we can see
that Japan controls the entirety of the Pacific Ocean and most of the Asian continent , including China , India and half of Russia . Japan is also shown to
control Australia , New Zealand , western Canada , Alaska and , all pacific states in the USA . We also see that that the Japanese Empire controls the Yucatán
Peninsula and South American countries like Chile . In contrast , Nazi Germany is shown as controlling all of Europe , Africa and the Middle East .
</item 1>

<item 2>
The Man in the High Castle
Yakima Herald . Retrieved March 28 , 2017 . Jump up ^ Andreeva , Nellie ( July 24 , 2014 ) . `` Amazon Studios Adds Drama ' The Man In The High Castle ' ,
Comedy ' Just Add Magic ' To Pilot Slate '' . Deadline . Retrieved January 10 , 2016 . Jump up ^ `` The Man in the High Castle : Season 1 , Episode 1 '' .
Amazon.com . Retrieved January 17 , 2015 . Jump up ^ `` The Man in the High Castle '' . Internet Movie Database . Retrieved January 18 , 2015 . Jump up ^ Lewis
, Hilary ( February 18 , 2015 ) . `` Amazon Orders 5 New Series Including ' Man in the High Castle ' '' . The Hollywood Reporter . Retrieved December 10 , 2015
. Jump up ^ Robertson , Adi ( February 18 , 2015 ) . `` Amazon green - lights The Man in the High Castle TV series '' . The Verge . Retrieved December 10 , 2015
.
</item 2>

<item 3>
The Man in the High Castle
The Nazis have drained the Mediterranean to make room for farmland , developed and used the hydrogen bomb , and designed rockets for extremely fast travel
across the world as well as space , having colonized the Moon , Venus , and Mars . The novel is set mostly in San Francisco in the P.S.A. ; here , Chinese
residents first appear in the novel as second - class citizens and black people as slaves . The secondary setting of the novel is the Rocky Mountains States ,
namely the cities of Cañon City , Denver and Cheyenne . Plot summary ( edit ) In 1962 , fifteen years after Imperial Japan and Nazi Germany have won World War
II , Robert `` Bob '' Childan owns an Americana antique shop in San Francisco , California ( located in the Japanese - occupied Pacific States of America ) ,
which is most commonly frequented by the Japanese , who make a fetish of romanticized American cultural artifacts . Childan is contacted by Nobusuke Tagomi , a
high - ranking Japanese trade official , who is seeking a gift to impress a visiting Swedish industrialist named Baynes .
</item 3>

<item 4>
The Man in the High Castle
The pilot episode was released by Amazon Studios on January 15 , 2015 , and was Amazon 's `` most watched pilot ever '' according to Amazon Studios ' vice
president , Roy Price . On February 18 , 2015 , Amazon green - lit the series . The show became available for streaming on November 20 , 2015 . The television
series diverges from the novel in many significant respects . Both the Pacific States of America and the Eastern American puppet state appear to be mere
provinces of the Japanese and German empires without any apparent autonomous ( even quisling ) government institutions whatsoever . The Rocky Mountain States
become an anarchic Neutral Zone . World War II appears to have ended following the destruction of Washington , D.C. with an atomic bomb , rather than a land
invasion as in the book . As for Hitler , while elderly , he is apparently mostly hale in his Season 1 finale appearance , though other characters elsewhere in
the season do reference his supposed physical infirmity .
</item 4>

<item 5>
The Man in the High Castle
In the novel , the Italian Empire is a minor power that controls North Africa and the Middle East ; in the series , however , it is shown through maps that
these territories are part of the Nazi Empire , suggesting that either the Italian Empire was annexed after the war or is self - governing within the Reich .
Characters from the book that do appear are in most cases far more fleshed out with deeper and sometimes rather different backstories than their novel originals
. For instance , Wegener is a Standartenführer in the SS , rather than a naval captain ( and oddly there are no German military or naval -- as opposed to SS --
personnel depicted anywhere in the first season ) . Rather than being a member of an organized internal resistance ( and despite his relatively low rank )
Wegener is a close personal confidante of Hitler and his disillusionment with the regime appears to be largely personal . Juliana and Frank are unmarried , but
living together , rather than divorced and separated . Frank has a sister , nephew and niece , although they are killed early in the series , and this propels
him into a more active role in relation to the resistance .
</item 5>

<item 6>
The Man in the High Castle
Soon , Joseph Goebbels is announced as the new German Chancellor . Baynes and Tagomi finally meet their Japanese contact as two agents of the Nazi secret police
, the Sicherheitsdienst ( SD ) , close in to arrest Baynes , who is actually revealed to be a Nazi defector named Rudolf Wegener . Wegener warns his contact , a
famed Japanese general , of Operation Dandelion , an upcoming Goebbels - approved plan for the Nazis to surprise - attack the Japanese Home Islands , in order
to obliterate them in one swift stroke . As Frink is elsewhere exposed as a Jew and arrested , Wegener and Tagomi are confronted by the SD agents , both of whom
Tagomi shoots dead with an antique American pistol . Back in Colorado , Joe abruptly changes his appearance and mannerisms before the trip to the High Castle ,
leading Juliana to deduce that he intends to actually murder Abendsen . Joe confirms this , revealing himself to be an undercover Swiss Nazi assassin . Juliana
mortally wounds Joe and drives off to warn Abendsen of the threat to his life .
</item 6>

<item 7>
The Man in the High Castle
Contents ( hide ) 1 Synopsis 1.1 Background 1.2 Plot summary 2 The Grasshopper Lies Heavy 3 Inspirations 4 Reception 5 Adaptations 5.1 Audiobook 5.2 Television
6 Incomplete sequel 7 See also 8 References 9 Further reading 10 External links Synopsis ( edit ) Fictional map of the world after the Axis victory Briefly ,
The Man in the High Castle is a `` fictional picture of a world divided by Germany and Japan , victors of the second World War '' . Background ( edit ) In the
novel 's alternate history , U.S. President Franklin D. Roosevelt was assassinated by Giuseppe Zangara in 1934 leading to the continuation of the Great
Depression and U.S. isolationism during the opening of World War II . Adolf Hitler led Nazi Germany to conquer most of Europe and the Soviet Union , murdering
Jews , Roma , Slavs , and other groups . Meanwhile , Imperial Japan occupied China , before taking control of India and Oceania . The Nazis then helped Italy
conquer most of Africa . As Japan invaded the U.S.
</item 7>

<item 8>
The Man in the High Castle
Wegener flies back to Germany where upon landing learns that Reinhard Heydrich ( a member of the anti-Dandelion faction ) has launched a coup against Goebbels ,
possibly installing himself as Chancellor . Meanwhile , Tagomi remains shaken by the shootout and goes to Childan to sell back the gun he used in the fight ;
however , instead , sensing the energy from one of Frink 's jewelery items , Tagomi impulsively buys it from Childan , before undergoing a spiritually intense
ambiguous moment where he momentarily perceives an alternative - history version of San Francisco . Later , Tagomi on a whim forces the German authorities to
release Frink , whom Tagomi has never met and does not know is the maker of the jewelery . Juliana soon has her own spiritual experience when she arrives in
Cheyenne . There , she discovers that Abendsen now lives in a normal house with his family , having left behind the High Castle due to a change of outlook ; he
no longer preoccupies himself with thoughts that he might soon be assassinated .
</item 8>

<item 9>
The Man in the High Castle
Frink reveals that the Wyndam - Matson Corporation has been supplying Childan with counterfeit antiques , which effectively works to blackmail Wyndam - Matson
for money to finance Frink 's new jewelry venture . Tagomi and Baynes meet , but Baynes repeatedly delays any real business as they await an expected third
party from Japan . Suddenly , the public receives news of the death of the recently ill Chancellor of Germany , Martin Bormann . Childan tentatively , on
consignment , takes some of Frink 's `` authentic '' new metalwork and attempts to curry favor with a Japanese client , who surprisingly considers Frink 's
jewellery immensely spiritually alive . Juliana and Joe take a road trip to Denver , Colorado , and Joe impulsively decides they should go on a side - trip to
meet the mysterious Hawthorne Abendsen , author of The Grasshopper Lies Heavy , who supposedly lives in a guarded fortress - like estate called the `` High
Castle '' in Cheyenne , Wyoming . Soon , Joseph Goebbels is announced as the new German Chancellor .
</item 9>

<item 10>
The Man in the High Castle
Childan 's store is stocked in part with antiques from the Wyndam - Matson Corporation , a metalworking company . Frank Frink ( formerly Fink ) , a secretly
Jewish -- American veteran of World War II , has just been fired from the Wyndam - Matson factory , when he agrees to join a former coworker to begin a
handcrafted jewelry business . Meanwhile , Frink 's ex-wife , Juliana , works as a judo instructor in Cañon City , Colorado ( in the neutral buffer zone of
Mountain States ) , where she begins a sexual relationship with an Italian truck driver and ex-soldier , Joe Cinnadella . Throughout the book , many of these
characters frequently make important decisions using prophetic messages they interpret from the I Ching . Many characters are also reading a widely banned yet
extremely popular new novel , The Grasshopper Lies Heavy , which depicts an alternate history in which the Allies won World War II , a concept that amazes and
intrigues its readers .
</item 10>
=== END ITEM id=bcb242e830d13a9b ===
