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

=== ITEM id=d157a7691f04b17c ===
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
doi : 10.1111 / cea. 12709 . PMID 26765082 ^ Jump up to : American Red Cross . `` Thunderstorm Safety Checklist '' ( PDF ) . American Red Cross . Retrieved 24
August 2011 . ^ Jump up to : National Weather Service Weather Forecast Office . `` Thunderstorm '' . Severe Weather Preparedness Information . Albuquerque , NM
: NOAA . Retrieved 24 August 2011 . Jump up ^ Federal Emergency Management Agency . `` Thunderstorms and Lightning '' . Ready . US Department of Homeland
Security . Archived from the original on 23 June 2011 . Retrieved 24 August 2011 . ^ Jump up to : Federal Emergency Management Agency . `` What to Do Before a
Thunderstorm '' . US Department of Homeland Security . Archived from the original on 20 August 2011 . Retrieved 24 August 2011 . ^ Jump up to : `` NWS Lightning
Safety Myths '' . Lightningsafety.noaa.gov. 2014 - 06 - 30 .
</item 2>

<item 3>
Thunderstorm
Motion ( edit ) Thunderstorm line viewed in reflectivity ( dBZ ) on a plan position indicator radar display The two major ways thunderstorms move are via
advection of the wind and propagation along outflow boundaries towards sources of greater heat and moisture . Many thunderstorms move with the mean wind speed
through the Earth 's troposphere , the lowest 8 kilometres ( 5.0 mi ) of the Earth 's atmosphere . Weaker thunderstorms are steered by winds closer to the Earth
's surface than stronger thunderstorms , as the weaker thunderstorms are not as tall . Organized , long - lived thunderstorm cells and complexes move at a right
angle to the direction of the vertical wind shear vector . If the gust front , or leading edge of the outflow boundary , races ahead of the thunderstorm , its
motion will accelerate in tandem . This is more of a factor with thunderstorms with heavy precipitation ( HP ) than with thunderstorms with low precipitation (
LP ) . When thunderstorms merge , which is most likely when numerous thunderstorms exist in proximity to each other , the motion of the stronger thunderstorm
normally dictates the future motion of the merged cell .
</item 3>

<item 4>
Thunderstorm
Lightningsafety.noaa.gov. 2014 - 06 - 30 . Retrieved 2014 - 08 - 20 . Jump up ^ `` NWS JetStream -- Lightning Frequently Asked Questions '' . Srh.noaa.gov. 2014
- 06 - 28 . Retrieved 2014 - 08 - 20 . Jump up ^ `` You 're not safer crouching : Six things you did n't know about lightning '' . LA Times . Retrieved 2014 -
08 - 20 . Jump up ^ National Geographic Almanac of Geography , ISBN 0 - 7922 - 3877 - X , page 75 . Jump up ^ `` How many thunderstorms occur each year ? '' .
Thunderstorms . Sky Fire Productions . Archived from the original on 2007 - 07 - 11 . Retrieved 2006 - 08 - 23 . Jump up ^ National Weather Service JetStream (
2008 - 10 - 08 ) . `` Tropical Cyclone Hazards '' . National Weather Service Southern Region Headquarters .
</item 4>

<item 5>
Thunderstorm
The National Weather Service ( NWS ) in the United States recommends several precautions that people should take if thunderstorms are likely to occur : Know the
names of local counties , cities , and towns , as these are how warnings are described . Monitor forecasts and weather conditions and know whether thunderstorms
are likely in the area . Be alert for natural signs of an approaching storm . Cancel or reschedule outdoor events ( to avoid being caught outdoors when a storm
hits ) . Take action early so you have time to get to a safe place . Get inside a substantial building or hard - topped metal vehicle before threatening weather
arrives . If you hear thunder , get to the safe place immediately . Avoid open areas like hilltops , fields , and beaches , and do n't be or be near the tallest
objects in an area when thunderstorms are occurring . Do n't shelter under tall or isolated trees during thunderstorms . If in the woods , put as much distance
between you and any trees during thunderstorms . If in a group , spread out so that you increase the chances for survivors who could come to the aid of any
victims from a lightning strike .
</item 5>

<item 6>
Thunderstorm
Safety ( edit ) While safety and preparedness often overlap , `` thunderstorm safety '' generally refers to what people should do during and after a storm . The
American Red Cross recommends that people follow these precautions if a storm is imminent or in progress : Take action immediately upon hearing thunder . Anyone
close enough to the storm to hear thunder can be struck by lightning . Avoid electrical appliances , including corded telephones . Cordless and wireless
telephones are safe to use during a thunderstorm . Close and stay away from windows and doors , as glass can become a serious hazard in high wind . Do not bathe
or shower , as plumbing conducts electricity . If driving , safely exit the roadway , turn on hazard lights , and park . Remain in the vehicle and avoid
touching metal . The NWS stopped recommending the `` lightning crouch '' in 2008 as it does n't provide a significant level of protection and will not
significantly lower the risk of being killed or injured from a nearby lightning strike . Frequent occurrences ( edit ) See also : United States rainfall
climatology A mild thunderstorm over Niagara Falls , Ontario .
</item 6>

<item 7>
Thunderstorm
Radio pulses produced by cosmic rays are being used to study how electric charges develop within thunderstorms . More organized meteorological projects such as
VORTEX2 use an array of sensors , such as the Doppler on Wheels , vehicles with mounted automated weather stations , weather balloons , and unmanned aircraft to
investigate thunderstorms expected to produce severe weather . Lightning is detected remotely using sensors that detect cloud - to - ground lightning strokes
with 95 percent accuracy in detection and within 250 metres ( 820 ft ) of their point of origin . Mythology and religion ( edit ) Thunderstorms strongly
influenced many early civilizations . Greeks believed that they were battles waged by Zeus , who hurled lightning bolts forged by Hephaestus . Some American
Indian tribes associated thunderstorms with the Thunderbird , who they believed was a servant of the Great Spirit . The Norse considered thunderstorms to occur
when Thor went to fight Jötnar , with the thunder and lightning being the effect of his strikes with the hammer Mjölnir . Hinduism recognizes Indra as the god
of rain and thunderstorms .
</item 7>

<item 8>
Thunderstorm
Retrieved 2009 - 09 - 09 . Jump up ^ Charles H. Paxton ; J. Colson & N. Carlisle ( 2008 ) . `` P2. 13 Florida lightning deaths and injuries 2004 -- 2007 '' .
American Meteorological Society . Retrieved 2009 - 09 - 05 . Jump up ^ G.E. Likens ; W.C. Keene ; J.M. Miller & J.N. Galloway ( 1987 ) . `` Chemistry of
precipitation from a remote , terrestrial site in Australia '' . Journal of Geophysical Research . 92 ( 13 ) : 299 -- 314 . Bibcode : 1987JGR ... 92 ... 299R .
doi : 10.1029 / JA092iA01p00299 . Jump up ^ Joel S. Levine ; Tommy R. Augustsson ; Iris C. Andersont ; James M. Hoell Jr . & Dana A. Brewer ( 1984 ) . ``
Tropospheric sources of NOx : Lightning and biology '' . Atmospheric Environment.
</item 8>

<item 9>
Thunderstorm
Archived from the original on 2009 - 06 - 21 . Retrieved 2009 - 08 - 28 . ^ Jump up to : John E. Oliver ( 2005 ) . Encyclopedia of World Climatology . Springer
. p. 401 . ISBN 978 - 1 - 4020 - 3264 - 6 . Retrieved 2009 - 08 - 28 . Jump up ^ Dongxia Liu ; Guili Feng & Shujun Wu ( February 2009 ) . `` The characteristics
of cloud - to - ground lightning activity in hailstorms over northern China '' . Atmospheric Research . 91 ( 2 -- 4 ) : 459 -- 465 . Bibcode : 2009AtmRe ... 91
... 459L . doi : 10.1016 / j. atmosres. 2008.06. 016 . Jump up ^ Damir Počakal ; Željko Večenaj & Janez Štalec ( 2009 ) .
</item 9>

<item 10>
Thunderstorm
The stronger the mean wind , the less likely other processes will be involved in storm motion . On weather radar , storms are tracked by using a prominent
feature and tracking it from scan to scan . Back - building thunderstorm ( edit ) A back - building thunderstorm , commonly referred to as a training
thunderstorm , is a thunderstorm in which new development takes place on the upwind side ( usually the west or southwest side in the Northern Hemisphere ) ,
such that the storm seems to remain stationary or propagate in a backward direction . Though the storm often appears stationary on radar , or even moving upwind
, this is an illusion . The storm is really a multi-cell storm with new , more vigorous cells that form on the upwind side , replacing older cells that continue
to drift downwind . When this happens , catastrophic flooding is possible . In Rapid City , South Dakota , in 1972 , an unusual alignment of winds at various
levels of the atmosphere combined to produce a continuously training set of cells that dropped an enormous quantity of rain upon the same area , resulting in
devastating flash flooding . A similar event occurred in Boscastle , England , on 16 August 2004 .
</item 10>
=== END ITEM id=d157a7691f04b17c ===

=== ITEM id=0b6107947b13c0ae ===
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
Will Friedle
Kiel ( 28 January 2009 ) . `` The Bravest & Boldest : Will Friedle '' . Comic Book Resources . Retrieved 30 November 2010 . Jump up ^ Andy Swift . `` ' Girl
Meets World ' : Eric Returns With Mr. Feeny -- Season 2 Spoilers - TVLine '' . TVLine . Retrieved 4 April 2016 . Jump up ^ `` Girl Meets Writers '' . Twitter .
Retrieved 4 April 2016 . Jump up ^ `` Will Friedle on Twitter '' . Twitter . Retrieved 4 April 2016 . Jump up ^ `` Art - World Movie Shot In State To Be Shown
Saturday In New Milford '' . tribunedigital - thecourant . Retrieved 4 April 2016 . Jump up ^ `` The Hartford Courant article archive - Friedle - Leary '' .
Retrieved 4 April 2016 . Jump up ^ `` The Hartford Courant article archive - She Chose to Have Both '' . Retrieved 4 April 2016 .
</item 1>

<item 2>
Will Friedle
One of his most recent roles has been the speaking voice of Bumblebee in the final episode of Transformers : Prime and the movie Transformers Prime Beast
Hunters : Predacons Rising and in the sequel series Transformers : Robots in Disguise , as well as in Transformers : Rescue Bots . In December 2013 , it was
announced that Friedle would reprise his Boy Meets World role of Eric Matthews on the continuation series Girl Meets World . Contents ( hide ) 1 Early life 2
Career 3 Personal life 4 Filmography 5 References 6 External links Early life ( edit ) Friedle was born in Hartford , Connecticut , the son of Patricia Joan (
née Leary ) and Gary Allen Friedle , both lawyers . After committing himself to becoming an actor , Friedle continually commuted from Avon , Connecticut to New
York City for auditions . He attended Avon High School in 1994 . Friedle attended Occidental College . In the July 4 , 2006 issue of People , Friedle mentioned
he had thought about attending culinary school .
</item 2>

<item 3>
Will Friedle
Personal life ( edit ) In 1997 , Friedle dated American actress Jennifer Love Hewitt , and co-starred with her in the romantic teen comedy Trojan War . Hewitt
made a guest appearance as Jennifer Love `` Feffy '' Fefferman on Boy Meets World episode `` And Then There Was Shawn '' . Friedle is best friends with actor
Jason Marsden and was the best man at Marsden 's wedding in October 2004 . The two have also worked together in numerous productions : Boy Meets World , Trojan
War , Static Shock , Kim Possible , Batman Beyond , Justice League Unlimited , Batman : The Brave and the Bold , ThunderCats and Mad . He married his girlfriend
, Susan Martens , on September 25 , 2016 in Connecticut . Filmography ( edit ) Film Year Film Role Notes Trojan War Brad Kimble 2000 Batman Beyond : Return of
the Joker Terry McGinnis / Batman Voice role Direct - to - DVD release 2003 National Lampoon 's Gold Diggers Calvin Menhoffer The Greatest Short Film Ever ! ! !
</item 3>

<item 4>
Harland Williams
Headmistress Tucker 1999 Superstar Slater 2000 The Whole Nine Yards Special Agent Steve Hanson Big Money Hustlas Officer Harry Cox Becoming Dick Richard Breggs
Freddy Got Fingered Darren Gary & Mike Mike Bonner ( voice ) 2002 Back by Midnight Sheriff Hubbard Sorority Boys Doofer / Roberta 2003 Family Tree Jake Kart
Racer Zee 2005 Lucky 13 Bleckman Because of Winn - Dixie Policeman Robots Lugnut ( voice ) 2006 Employee of the Month Russell Porpis - Gunders Surf School Rip
Ergo Proxy Head Dog Cop ( voice ) 1 episode 2007 Meet the Robinsons Carl ( voice ) The Dukes of Hazzard : The Beginning Sheriff Rosco P. Coltrane 2008 Bachelor
Party 2 : The Last Temptation Derek Madagascar : Escape 2 Africa Giraffe ( voice ) 2009 Fudgy Wudgy Fudge Face Elmore P.
</item 4>

<item 5>
Will Friedle
Career ( edit ) After co-hosting Nickelodeon 's Do n't Just Sit There for three seasons ( 1988 -- 1991 ) , Friedle was cast as Cory Matthews ' likeable but
underachieving elder brother Eric on the long - running TV sitcom Boy Meets World from 1993 to 2000 . In 2004 , he co-starred with Chris Owen in the independent
film National Lampoon 's Gold Diggers ( originally titled Lady Killers ) . He also appeared in a supporting role in the 2005 ABC Family movie Everything You
Want . Friedle was cast as Mike in the WB sitcom Off Centre , but could not accept the role because of a previous commitment to the UPN series The Random Years
. The role ended up going to Eddie Kaye Thomas . Friedle voiced Seifer Almasy in the English version of the video game Kingdom Hearts II , Gideon Wyeth in the
English version of the video game Advent Rising , and Jaster Rogue in the English version of the video game Rogue Galaxy .
</item 5>

<item 6>
Will Friedle
! ! Mark Short film Howl 's Moving Castle Additional voices Voice role 2005 Everything You Want Calvin Dillwaller 2008 Batman : Gotham Knight Anton , Cultist ,
Youth # 1 Voice role Direct - to - DVD release 2015 Batman Unlimited : Animal Instincts Nightwing / Dick Grayson Voice role Direct - to - DVD release Batman
Unlimited : Monster Mayhem Nightwing / Dick Grayson Voice role Direct - to - DVD release 2016 Lego DC Comics Super Heroes : Justice League : Gotham City
Breakout Nightwing / Dick Grayson Voice role Direct - to - DVD release Batman Unlimited : Mech vs. Mutants Nightwing / Dick Grayson Voice role Direct - to - DVD
release Television Year Title Role Notes Do n't Just Sit There Host Law & Order Russ Episode : `` Trust '' 1993 -- 2000 Boy Meets World Eric Matthews Series
regular ( 157 episodes ) Young Artist Award ( nominated ) The Gift of Love Luke CBS television movie Are You Afraid of the Dark ?
</item 6>

<item 7>
Will Friedle
Retrieved 4 April 2016 . Jump up ^ Friedle lands UPN sitcom Neil Wilkes , DigitalSpy , 4 March 2002 Jump up ^ `` Will Friedle '' . Retrieved 4 April 2016 . Jump
up ^ `` Boy Meets World star Will Friedle celebrates his marriage '' . Retrieved 15 August 2017 . Jump up ^ `` Gary Miereanu '' . Twitter . Retrieved 4 April
2016 . Jump up ^ Gerding , Stephen ( 2 June 2015 ) . `` EXCLUSIVE TRAILER DEBUT : Joker Rules in `` Batman Unlimited : Monster Mayhem '' `` . Comic Book
Resources . Retrieved 6 June 2015 . Jump up ^ Siegel , Lucas ( April 4 , 2016 ) . `` EXCLUSIVE : WB Reveals Next LEGO DC Comics Movie with Announcement Trailer
'' . Comic Book Resources . Retrieved July 25 , 2016 . Jump up ^ `` New Frontiers '' . Avengers Assemble . Season 2 . Episode 25 . September 13 , 2015 .
</item 7>

<item 8>
Daren Kagasoff
The film was released in October 2014 . In September 2014 , Kagasoff joined the cast of the Red Band Society , a comedy - drama series on Fox for the 2014 -- 15
American television season developed by Margaret Nagle . He had a recurring role playing Kara 's love interest , Hunter Cole . The series premiered on September
17 , 2014 and was canceled after one season . Filmography ( edit ) Television and film roles Year Title Role Notes 2008 -- 2013 Secret Life of the American
Teenager , The The Secret Life of the American Teenager Richard `` Ricky '' Underwood Main role 2014 Ouija Trevor Film 2014 Blue Daren Web series ; 2 episodes
2014 Delirium Alex Sheathes Unsold television pilot ( FOX ) 2014 Stalker Eric Bates Episode : Pilot 2014 -- 2015 Red Band Society Hunter Cole Recurring role , 7
episodes 2015 Paradise Pictures Wyatt Osborne Unsold television pilot ( USA ) 2017 S.W.A.T.
</item 8>

<item 9>
Will Friedle
Will Friedle - wikipedia Will Friedle Jump to : navigation , search Will Friedle William Alan Friedle ( 1976 - 08 - 11 ) August 11 , 1976 ( age 41 ) Hartford ,
Connecticut , U.S. Occupation Actor , voice actor Years active 1988 -- present Spouse ( s ) Susan Martens ( m . 2016 ) William Alan `` Will '' Friedle ( born
August 11 , 1976 ) is an American actor and voice actor . He is best known for his comedic roles , most notably the underachieving elder brother Eric Matthews
on the long - running TV sitcom Boy Meets World from 1993 to 2000 . More recently , he has voiced a number of animated characters such as Terry McGinnis /
Batman , the title character of Batman Beyond , and Ron Stoppable of Kim Possible . He voices Deadpool in Ultimate Spider - Man and Star - Lord in the animated
version of Guardians of the Galaxy , replacing Chris Cox . He also performed the voices of Doyle in The Secret Saturdays , Lion - O in the rebooted ThunderCats
series , and Blue Beetle on Batman : The Brave and the Bold .
</item 9>

<item 10>
Will Friedle
Jimmy Armstrong Episode : `` The Tale of the Long Ago Locket '' ABC Afterschool Special Jason Gallagher Episode : `` Educating Mom '' 1998 My Date with the
President 's Daughter Duncan Fletcher Disney Channel television movie 1999 Zoe , Duncan , Jack and Jane Jeremy Pinter Episode : `` A Good Man is Hard to Find ''
H-E Double Hockey Sticks Griffelkin ABC television movie Odd Man Out Phillip Evans Episode : `` Punch Line '' 1999 -- 2001 Batman Beyond Terry McGinnis / Batman
, additional voices Voice role 48 episodes The Zeta Project Terry McGinnis / Batman Voice role Episode : `` Shadows '' 3rd Rock from the Sun Stan Episode : ``
Dick Soup for the Soul '' Go Fish Pete Troutner Series regular 2002 All That Himself Season 7 ,
</item 10>
=== END ITEM id=0b6107947b13c0ae ===

=== ITEM id=908cbee1a368950c ===
Gold sentences and facts:
Sentence 0: Queen Elizabeth II is the sovereign , and her heir apparent is her eldest son , Charles , Prince of Wales . Next in line after him is Prince William , Duke of Cambridge , the Prince of Wales 's elder son . Third in line is Prince George , the eldest child of the Duke of Cambridge , followed by his sister , Princess Charlotte and younger brother , Prince Louis . Sixth in line is Prince Henry of Wales , the younger son of the Prince of Wales . Any of the first six in line marrying without the sovereign 's consent would be disqualified from succession .
  Fact 0.0: Queen Elizabeth II is the sovereign.
  Fact 0.1: Charles, Prince of Wales, is the heir apparent to the sovereign, Queen Elizabeth II.
  Fact 0.2: Charles, Prince of Wales, is the eldest son of Queen Elizabeth II.
  Fact 0.3: Prince William, Duke of Cambridge, is next in line for succession after Charles, Prince of Wales.
  Fact 0.4: Prince William is the elder son of the Prince of Wales.
  Fact 0.5: Prince George is third in line for succession.
  Fact 0.6: Prince George is the eldest child of the Duke of Cambridge.
  Fact 0.7: Princess Charlotte follows Prince George in the line of succession.
  Fact 0.8: Princess Charlotte is the sister of Prince George.
  Fact 0.9: Prince Louis is the younger brother of Prince George.
  Fact 0.10: Prince Henry of Wales is sixth in line for succession.
  Fact 0.11: Prince Henry of Wales is the younger son of the Prince of Wales.
  Fact 0.12: Any of the first six in line to the throne who marries without the sovereign's consent would be disqualified from succession.

Retrieved items:
<item 1>
Succession to the British throne
After Mary II died in 1694 , her husband continued to reign alone until his own death in 1702 . The line of succession provided for by the Bill of Rights was
almost at an end ; William and Mary never had any children , and Princess Anne 's children had all died . Therefore , Parliament passed the Act of Settlement .
The Act maintained the provision of the Bill of Rights whereby William would be succeeded by Princess Anne and her descendants , and thereafter by his own
descendants from future marriages . The Act , however , declared that they would be followed by James I & VI 's granddaughter Sophia , Electress Dowager of
Hanover ( the daughter of James 's daughter Elizabeth Stuart ) , and her heirs . As under the Bill of Rights , non-Protestants and those who married Roman
Catholics were excluded . Electress Sophia of Hanover ( 1630 -- 1714 ) Upon William 's death , Anne became Queen of England , Scotland and Ireland . Because the
Parliament of England settled on Sophia as Anne 's heir - presumptive without consulting Scottish leaders , the Estates of Scotland retaliated by passing the
Scottish Act of Security .
</item 1>

<item 2>
Succession to the British throne
Succession to the British throne - wikipedia Succession to the British throne This is the latest accepted revision , reviewed on 27 April 2018 . Jump to :
navigation , search The Sovereign 's Throne in the House of Lords , from which the speech is delivered at State Openings of Parliament Succession to the British
throne is determined by descent , gender ( for people born before October 2011 ) , legitimacy , and religion . Under common law , the Crown is inherited by a
sovereign 's children or by a childless sovereign 's nearest collateral line . The Bill of Rights 1689 and the Act of Settlement 1701 , restrict succession to
the throne to the legitimate Protestant descendants of Sophia of Hanover that are in `` communion with the Church of England '' . Spouses of Roman Catholics
were disqualified from 1689 until the law was amended in 2015 . Protestant descendants of those excluded for being Roman Catholics are eligible . Queen
Elizabeth II is the sovereign , and her heir apparent is her eldest son , Charles , Prince of Wales . Next in line after him is Prince William , Duke of
Cambridge , the Prince of Wales 's elder son .
</item 2>

<item 3>
Succession to the British throne
Edward died childless in 1972 . Edward 's abdication was `` a demise of the Crown '' ( in the words of the Act ) , and the Duke of York , his brother who was
then next in the line , immediately succeeded to the throne and to its `` rights , privileges , and dignities '' , taking the regnal name George VI . He in turn
was succeeded in 1952 by his own elder daughter , Elizabeth II . By that time the monarch of the United Kingdom no longer reigned in the greater part of Ireland
( which had become a republic in 1949 ) , but was the monarch of a number of independent sovereign states ( Commonwealth realms ) . Commonwealth realms ( edit )
Further information : Perth Agreement Elizabeth II at her coronation in 1953 , passing to the left of the Coronation Chair Each of the Commonwealth realms has
the same person as monarch and , to maintain that arrangement , they have agreed to continue the same line of succession ; some realms do so through domestic
succession laws , while others stipulate whoever is monarch of the United Kingdom will also be monarch of that realm .
</item 3>

<item 4>
Succession to the British throne
Each of those countries has the same person as monarch and the same order of succession . In 2011 , the prime ministers of the realms agreed unanimously to
adopt a common approach to amending the rules on the succession to their respective Crowns so that absolute primogeniture would apply for persons born after the
date of the agreement , instead of male - preference primogeniture , and the ban on marriages to Roman Catholics would be lifted , but the monarch would still
need to be in communion with the Church of England . After the necessary legislation had been enacted in accordance with each realm 's constitution , the
changes took effect on 26 March 2015 . Contents ( hide ) 1 Current line of succession 2 History 2.1 England 2.2 Scotland 2.3 After the union of the crowns 2.3.
1 Stuarts 2.3.
</item 4>

<item 5>
Succession to the British throne
Jane was also originally excluded on the premise that no woman could reign over England . Nonetheless , the will , which originally referred to Jane 's heirs -
male , was amended to refer to Jane and her heirs - male . Upon Edward VI 's death in 1553 , Jane was proclaimed Queen of England and Ireland . She was not
universally recognised and after nine days she was overthrown by the popular Mary . As Henry VIII 's will had been approved by an Act of Parliament in 1544 ,
Edward 's contravening will was unlawful and ignored . Mary was succeeded by her half - sister , Elizabeth , who broke with the precedents of many of her
predecessors , and refused to name an heir . Whilst previous monarchs ( including Henry VIII ) had specifically been granted authority to settle uncertain
successions in their wills , the Treasons Act 1571 asserted that Parliament had the right to settle disputes , and made it treason to deny Parliamentary
authority . Wary of threats from other possible heirs , Parliament further passed the Act of Association 1584 , which provided that any individual involved in
attempts to murder the Sovereign would be disqualified from succeeding .
</item 5>

<item 6>
Succession to the British throne
Though the two remained illegitimate , an Act of Parliament passed in 1544 allowed reinserting them , providing further `` that the King should and might give ,
will , limit , assign , appoint or dispose the said imperial Crown and the other premises ... by letters patent or last will in writing . '' Mary and Elizabeth
, under Henry VIII 's will , were to be followed by descendants of the King 's deceased sister Mary Tudor , Duchess of Suffolk ( he however excluded his niece
Frances Brandon , Duchess of Suffolk ) . This will also excluded from the succession the descendants of Henry 's eldest sister Margaret Tudor , who were the
rulers of Scotland . Queen Elizabeth I enthroned in Parliament When Henry VIII died in 1547 , the young Edward succeeded him , becoming Edward VI . Edward VI
was the first Protestant Sovereign to succeed to the rule of England . He attempted to divert the course of succession in his will to prevent his Catholic half
- sister , Mary , from inheriting the throne . He excluded Mary and Elizabeth , settling on the Duchess of Suffolk 's daughter , Lady Jane Grey . Jane was also
originally excluded on the premise that no woman could reign over England .
</item 6>

<item 7>
Under the 1702 and 1703 Treason Acts, attempting to prevent the next heir from succeeding is treason, punishable by life imprisonment since 1998. In
Commonwealth realms, succession occurs immediately upon the sovereign’s death. The Accession Council then proclaims the new monarch, traditionally in public
places including London, York and Edinburgh. Accession Day is observed throughout the reign, with royal gun salutes in the UK. James I’s accession was notably
proclaimed by a council after Elizabeth I’s
</item 7>

<item 8>
Succession to the British throne
In February 1952 , on her accession , Elizabeth II was proclaimed as sovereign separately throughout her realms . In October 2011 , the heads of government of
all 16 realms agreed unanimously at a meeting held in Perth , Australia , to proposed changes to the royal succession laws that would see the order of
succession for people born after 28 October 2011 governed by absolute primogeniture -- wherein succession passes to an individual 's children according to birth
order , regardless of gender -- instead of male - preference primogeniture . They also agreed to lift a ban on those who marry Roman Catholics . The ban on
Catholics themselves was retained to ensure that the monarch would be in communion with the Church of England . The changes came into effect across the
Commonwealth realms on 26 March 2015 , after legislation was made according to each realm 's constitution . Following the changes coming into effect , the
positions of the first 27 in line remained unchanged , including Princess Anne and her children and grandchildren , until the birth of Princess Charlotte of
Cambridge on 2 May 2015 .
</item 8>

<item 9>
The British throne passes to legitimate Protestant descendants of Sophia of Hanover; Catholics remain excluded, though marrying a Catholic no longer
disqualifies an heir. Reforms agreed by the 16 Commonwealth realms took effect in 2015, ending male preference for those born after 28 October 2011 and
requiring sovereign consent for the first six heirs’ marriages. In April 2018, the first six were Charles, William, George, Charlotte, Louis and Harry.
Succession law evolved through Parliament,
</item 9>

<item 10>
Succession to the British throne
3 : `` A person who ( when the person marries ) is one of the 6 persons next in the line of succession to the Crown must obtain the consent of Her Majesty
before marrying . '' Jump up ^ Most sources , including Whitaker 's Almanack , exclude her as a confirmed Catholic , but Debrett 's includes her in line . Jump
up ^ Albert and Leopold Windsor were listed on The Official Website of the British Monarchy until 2015 and in the 2013 edition of Whitaker 's Almanack as
following Estella Taylor ( b 2004 ) and eligible to succeed ; MSN News , Debrett 's and Whitaker 's Almanack 2015 lists them after Lady Amelia Windsor and
before Lady Helen Taylor . They were baptised as Catholics , and are not listed in line in editions of Whitaker 's earlier than 2012 . Jump up ^ Neither the
Legitimacy Act 1926 nor the Legitimacy Act 1959 changed the Law of Succession . Jump up ^ Per the His Majesty 's Declaration of Abdication Act 1936 .
</item 10>
=== END ITEM id=908cbee1a368950c ===

=== ITEM id=8dc8119c6f0ed0b5 ===
Gold sentences and facts:
Sentence 0: `` What a Friend We Have in Jesus '' is a Christian hymn originally written by Joseph M. Scriven as a poem in 1855 to comfort his mother who was living in Ireland while he was in Canada . Scriven originally published the poem anonymously , and only received full credit for it in the 1880s . The tune to the hymn was composed by Charles Crozat Converse in 1868 . William Bolcom composed a setting of the hymn .
  Fact 0.0: "What a Friend We Have in Jesus" is a Christian hymn.
  Fact 0.1: The hymn "What a Friend We Have in Jesus" was originally written by Joseph M. Scriven as a poem in 1855.
  Fact 0.2: Joseph M. Scriven wrote the poem to comfort his mother.
  Fact 0.3: Joseph M. Scriven's mother was living in Ireland while he was in Canada.
  Fact 0.4: Joseph M. Scriven originally published the poem anonymously.
  Fact 0.5: Joseph M. Scriven only received full credit for the poem in the 1880s.
  Fact 0.6: The tune of the hymn "What a Friend We Have in Jesus" was composed by Charles Crozat Converse in 1868.
  Fact 0.7: William Bolcom composed a setting of the hymn "What a Friend We Have in Jesus".

Retrieved items:
<item 1>
What a Friend We Have in Jesus
Glen Campbell ( 1989 , Favorite Hymns ) Driving Miss Daisy ( 1989 ) sung at Little Friendship Baptist Church in Atlanta , Georgia Ike & Tina Turner ( 1974 , The
Gospel According to Ike & Tina ) The John Tesh Project ( 2000 , Pure Hymns ) Amy Grant recorded it as part of the medley `` What a Friend We Have in Jesus / Old
Rugged Cross / How Great Thou Art '' on her 2002 studio album Legacy ... Hymns and Faith , and later included on her 2015 compilation album Be Still and Know
... Hymns & Faith . Alan Jackson ( 2006 , Precious Memories ) Brad Paisley ( 2008 , Play ) Ronnie Milsap ( 2009 , Then Sings My Soul ) Monty Alexander ( 2013 ,
Uplift 2 ) In asia ( edit ) In Japan , the hymn 's title was originally translated as `` Itsukushimi Fukaki '' ( いつくしみ 深き , lit . `` Merciful '' ) , which is
what it is best known by .
</item 1>

<item 2>
What a Friend We Have in Jesus
wikipedia.org/w/index.php?title=What_a_Friend_We_Have_in_Jesus&oldid=831327264 '' Categories : 1928 songs Canadian Christian hymns Columbia Records singles Glen
Campbell songs Mississippi John Hurt songs Pace Jubilee Singers songs Songs about Jesus Washington Phillips songs Hidden categories : Articles with hAudio
microformats Articles containing Japanese - language text Articles with trivia sections from February 2018 All articles with trivia sections Talk Contents About
Wikipedia Wikisource 日本 語 Norsk Svenska 中文 Edit links This page was last edited on 20 March 2018 , at 00 : 54 . Text is available under the Creative Commons
Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a
registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 2>

<item 3>
What a Friend We Have in Jesus
What a Friend We Have in Jesus - wikipedia What a Friend We Have in Jesus Jump to : navigation , search `` What a Friend We Have in Jesus '' Song Genre Hymn
Composer ( s ) Charles C. Converse ( 1868 ) Lyricist ( s ) Joseph M. Scriven ( 1855 ) `` What a Friend We Have in Jesus '' is a Christian hymn originally
written by Joseph M. Scriven as a poem in 1855 to comfort his mother who was living in Ireland while he was in Canada . Scriven originally published the poem
anonymously , and only received full credit for it in the 1880s . The tune to the hymn was composed by Charles Crozat Converse in 1868 . William Bolcom composed
a setting of the hymn . The hymn also has many versions with different lyrics in multiple languages . The Handbook to the Lutheran Hymnal notes , `` In spite of
the fact that this hymn , with its tune , has been criticized as being too much on the order of the sentimental gospel type , its popularity remains strong ,
and the hymn retains a place in modern hymnals . ''
</item 3>

<item 4>
What a Friend We Have in Jesus
External links ( edit ) Piano score ( pdf ) at Timeless Truths Text , MIDI , and piano score for `` What a Friend We Have in Jesus '' at HymnSite.
</item 4>

<item 5>
What a Friend We Have in Jesus
The tune was in the labor protest movement due to its wide popularity . `` Dump the Bosses Off Your Backs '' ( words by John Brill ) was published in the IWW ``
Little Red Songbook '' and has been recorded by dozens of artists including Utah Phillips , Anne Feeney , and Joe Glazer . A Norwegian version was used in the
film Trollhunter . Published versions ( edit ) Rise Up Singing , page 98 The Lutheran Hymnal , hymn # 457 References ( edit ) Wikisource has original text
related to this article : What a Friend We Have in Jesus Jump up ^ `` What a Friend We Have in Jesus '' . Retrieved 2007 - 05 - 03 . Jump up ^ Polack , W.G. (
1941 ) . Handbook to the Lutheran Hymnal . St. Louis : Concordia . p. 323 . Jump up ^ `` Washington Phillips discography '' . wirz.de . Retrieved August 24 ,
2015 . Jump up ^ George Washington Phillips : Jesus Is My Friend at AllMusic .
</item 5>

<item 6>
What a Friend We Have in Jesus
It is translated as `` Preethi gala mana Yesu '' and it is listed in the Andhra Kraisthava Keerthanalu ( Andhra Christian hymnals ) as hymn No. 407 . In India ,
the hymn sung in Konkani , `` Dulob Jezu '' , was translated by Felix Paul Noronha of Mangalore . In popular culture ( edit ) This section indiscriminately
collects miscellaneous information . Please compress this material to remove any irrelevant or unimportant information . ( February 2018 ) A version was sung
during World War I with the line `` When this bloody war is over '' in place of the original title . The tune was used in the 1969 film , Oh ! What a Lovely War
with the title `` When This Lousy War Is Over '' . The tune was used in Alan Price 's song Changes , which was written for the 1973 film O Lucky Man ! and later
became a hit when used in the 1987 Volkswagen Golf television commercial . The tune was in the labor protest movement due to its wide popularity .
</item 6>

<item 7>
What a Friend We Have in Jesus
Jump up ^ George Washington Phillips : Jesus Is My Friend at AllMusic . Retrieved August 24 , 2015 . Jump up ^ `` Volkswagen Golf -- Paula Hamilton -- 1987 --
UK Advert '' . Retro Commercials . March 13 , 2010 . Retrieved September 26 , 2011 . Jump up ^ `` Pure Hymns overview '' . Allmusic . Retrieved July 4 , 2010 .
Jump up ^ `` Monty Alexander UPLIFT 2 '' . Jazz Legacy Productions . Retrieved January 11 , 2014 . Jump up ^ `` Itsukushimi Fukaki song sheet '' ( PDF ) .
Retrieved 2007 - 07 - 16 . ^ Jump up to : `` 312 : What a Friend We Have in Jesus '' . Retrieved 2008 - 03 - 19 . Jump up ^ Christhiya Keerthanangal ( Christian
Hymns ) . 2007 . Hymn 173 . External links ( edit ) Piano score ( pdf ) at Timeless Truths Text , MIDI ,
</item 7>

<item 8>
What a Friend We Have in Jesus
`` Merciful '' ) , which is what it is best known by . In 1910 , Daisui Sugitani re-wrote the lyrics in Japanese and changed the title to `` Hoshi no Yo '' ( 星
の 界 , lit . `` World of Stars '' ) . Another version was written by Ryūkō Kawaji with the title `` Hoshi no Sekai '' ( 星 の 世界 , lit . `` World of Stars '' ) .
It is also known by the title `` Tsumitoga o Ninou '' ( つみ と が を に の う , lit . `` To Bear Sin '' ) . The hymn is popular at wedding ceremonies in Japan . In
Indonesia , the hymn is known as `` Yesus Kawan yang Sejati '' and is sung in Indonesian or Batak ( the indigenous language of North Sumatra ) in Manado ,
Maluccan , and Protestant churches ( around 6 % of the population ) .
</item 8>

<item 9>
What a Friend We Have in Jesus
In some settings , the lyrics have been matched to other tunes such as the Welsh `` Calon Lân '' ( originally wedded to the Welsh poem translated as `` A Pure
Heart '' ) . What a Friend We Have In Jesus Performed by the Onward Brass Band Problems playing this file ? See media help . Contents ( hide ) 1 Renditions 1.1
In Asia 2 In popular culture 3 Published versions 4 References 5 External links Renditions ( edit ) Washington Phillips , as `` Jesus Is My Friend '' ( 1928 ,
Columbia Records ) Bing Crosby ( 1951 , Beloved Hymns ) Rosemary Clooney ( 1959 , Hymns from the Heart , MGM Records ) Ella Fitzgerald with the Ralph Carmichael
Choir ( 1967 , Brighten the Corner ) Aretha Franklin ( 1972 , Amazing Grace ) Alan Price used the tune for his song `` Changes '' in the 1973 film O Lucky Man !
, which was reused in a Volkswagen commercial of the same name with Paula Hamilton in the 1980s .
</item 9>

<item 10>
What a Friend We Have in Jesus
and piano score for `` What a Friend We Have in Jesus '' at HymnSite.com `` What a Friend We Have in Jesus '' Mp3 and lyrics ( hide ) Washington Phillips
Original Columbia 78 rpm records ( chronological ) `` Lift Him Up That 's All '' / `` Take Your Burden to the Lord and Leave It There '' `` Denomination Blues
-- Part 1 '' / `` Denomination Blues -- Part 2 '' `` Mother 's Last Word to Her Son '' / `` Paul and Silas in Jail '' `` What Are They Doing in Heaven Today ''
/ `` Jesus Is My Friend '' `` I Am Born to Preach the Gospel '' / `` Train Your Child '' `` A Mother 's Last Word to Her Daughter '' / `` I 've Got the Key to
the Kingdom '' `` I Had a Good Father and Mother '' / `` The Church Needs Good Deacons '' Later song releases `` You Ca n't Stop a Tattler -- Part 1 '' `` You
Ca n't Stop a Tattler -- Part 2 '' Related articles List of songs ( with recording & catalogue details ) Retrieved from `` https://en.wikipedia.org/w/index.
</item 10>
=== END ITEM id=8dc8119c6f0ed0b5 ===

=== ITEM id=fae7ea1a661c2693 ===
Gold sentences and facts:
Sentence 0: On the Origin of Species ( or more completely , On the Origin of Species by Means of Natural Selection , or the Preservation of Favoured Races in the Struggle for Life ) , published on 24 November 1859 , is a work of scientific literature by Charles Darwin which is considered to be the foundation of evolutionary biology . Darwin 's book introduced the scientific theory that populations evolve over the course of generations through a process of natural selection . It presented a body of evidence that the diversity of life arose by common descent through a branching pattern of evolution . Darwin included evidence that he had gathered on the Beagle expedition in the 1830s and his subsequent findings from research , correspondence , and experimentation .
  Fact 0.0: The full title of On the Origin of Species is On the Origin of Species by Means of Natural Selection, or the Preservation of Favoured Races in the Struggle for Life.
  Fact 0.1: On the Origin of Species was published on 24 November 1859.
  Fact 0.2: On the Origin of Species is a work of scientific literature by Charles Darwin.
  Fact 0.3: On the Origin of Species is considered to be the foundation of evolutionary biology.
  Fact 0.4: Charles Darwin's book On the Origin of Species introduced the scientific theory that populations evolve over the course of generations.
  Fact 0.5: The scientific theory in On the Origin of Species holds that populations evolve through a process of natural selection.
  Fact 0.6: On the Origin of Species presented a body of evidence.
  Fact 0.7: The evidence in On the Origin of Species holds that the diversity of life arose by common descent.
  Fact 0.8: The evidence in On the Origin of Species holds that this common descent follows a branching pattern of evolution.
  Fact 0.9: Charles Darwin included evidence he had gathered on the Beagle expedition in the 1830s in On the Origin of Species.
  Fact 0.10: Charles Darwin included his subsequent findings from research in On the Origin of Species.
  Fact 0.11: Charles Darwin included his subsequent findings from correspondence in On the Origin of Species.
  Fact 0.12: Charles Darwin included his subsequent findings from experimentation in On the Origin of Species.
Sentence 1: On the Origin of Species The title page of the 1859 edition of On the Origin of Species Author Charles Darwin Country United Kingdom of Great Britain and Ireland Language English Subject Natural selection Evolutionary biology Published 24 November 1859 ( John Murray ) Media type Print ( Hardback ) Pages 502 OCLC 352242 Preceded by On the Tendency of Species to form Varieties ; and on the Perpetuation of Varieties and Species by Natural Means of Selection Followed by Fertilisation of Orchids
  Fact 1.0: On the Origin of Species has the author Charles Darwin.
  Fact 1.1: On the Origin of Species has the country United Kingdom of Great Britain and Ireland.
  Fact 1.2: On the Origin of Species is written in English.
  Fact 1.3: On the Origin of Species has the subject natural selection.
  Fact 1.4: On the Origin of Species has the subject evolutionary biology.
  Fact 1.5: On the Origin of Species was published on 24 November 1859 by John Murray.
  Fact 1.6: The title page of the 1859 edition of On the Origin of Species is shown.
  Fact 1.7: The media type of On the Origin of Species is print (hardback).
  Fact 1.8: On the Origin of Species has 502 pages.
  Fact 1.9: On the Origin of Species has OCLC number 352242.
  Fact 1.10: On the Origin of Species was preceded by On the Tendency of Species to form Varieties; and on the Perpetuation of Varieties and Species by Natural Means of Selection.
  Fact 1.11: On the Origin of Species was followed by Fertilisation of Orchids.

Retrieved items:
<item 1>
On the Origin of Species
The second edition of 3,000 copies was quickly brought out on 7 January 1860 , and incorporated numerous corrections as well as a response to religious
objections by the addition of a new epigraph on page ii , a quotation from Charles Kingsley , and the phrase `` by the Creator '' added to the closing sentence
. During Darwin 's lifetime the book went through six editions , with cumulative changes and revisions to deal with counter-arguments raised . The third edition
came out in 1861 , with a number of sentences rewritten or added and an introductory appendix , An Historical Sketch of the Recent Progress of Opinion on the
Origin of Species , while the fourth in 1866 had further revisions . The fifth edition , published on 10 February 1869 , incorporated more changes and for the
first time included the phrase `` survival of the fittest '' , which had been coined by the philosopher Herbert Spencer in his Principles of Biology ( 1864 ) .
In January 1871 , George Jackson Mivart 's On the Genesis of Species listed detailed arguments against natural selection , and claimed it included false
metaphysics .
</item 1>

<item 2>
On the Origin of Species
By 1977 , it had appeared in an additional 18 languages . Content ( edit ) Title pages and introduction ( edit ) John Gould 's illustration of Darwin 's rhea
was published in 1841 . The existence of two rhea species with overlapping ranges influenced Darwin . Page ii contains quotations by William Whewell and Francis
Bacon on the theology of natural laws , harmonising science and religion in accordance with Isaac Newton 's belief in a rational God who established a law -
abiding cosmos . In the second edition , Darwin added an epigraph from Joseph Butler affirming that God could work through scientific laws as much as through
miracles , in a nod to the religious concerns of his oldest friends . The Introduction establishes Darwin 's credentials as a naturalist and author , then
refers to John Herschel 's letter suggesting that the origin of species `` would be found to be a natural in contradistinction to a miraculous process '' : WHEN
on board HMS Beagle , as naturalist , I was much struck with certain facts in the distribution of the inhabitants of South America , and in the geological
relations of the present to the past inhabitants of that continent .
</item 2>

<item 3>
On the Origin of Species
The theory explains the diversity of living organisms and their adaptation to the environment . It makes sense of the geologic record , biogeography , parallels
in embryonic development , biological homologies , vestigiality , cladistics , phylogenetics and other fields , with unrivalled explanatory power ; it has also
become essential to applied sciences such as medicine and agriculture . Despite the scientific consensus , a religion - based political controversy has
developed over how evolution is taught in schools , especially in the United States . Interest in Darwin 's writings continues , and scholars have generated an
extensive literature , the Darwin Industry , about his life and work . The text of Origin itself has been subject to much analysis including a variorum ,
detailing the changes made in every edition , first published in 1959 , and a concordance , an exhaustive external index published in 1981 . Worldwide
commemorations of the 150th anniversary of the publication of On the Origin of Species and the bicentenary of Darwin 's birth were scheduled for 2009 . They
celebrated the ideas which `` over the last 150 years have revolutionised our understanding of nature and our place within it '' .
</item 3>

<item 4>
On the Origin of Species
Published anonymously . Huxley , Thomas ( 1863 ) , Six Lectures to Working Men `` On Our Knowledge of the Causes of the Phenomena of Organic Nature '' (
Republished in Volume II of his Collected Essays , Darwiniana ) , retrieved 15 December 2006 Keynes , Richard , ed. ( 2000 ) , Charles Darwin 's Zoology Notes &
Specimen Lists from HMS Beagle , Cambridge University Press , ISBN 0 - 521 - 67350 - X Kreeft , Peter ( 2001 ) , Catholic Christianity , San Francisco :
Ignatius Press , ISBN 0 - 89870 - 798 - 6 Larson , Edward J. ( 2004 ) , Evolution : The Remarkable History of a Scientific Theory , New York : Modern Library ,
ISBN 0 - 8129 - 6849 - 2 Leifchild ( 1859 ) , `` Review of ' Origin ' '' , Athenaeum ( No. 1673 , 19 November 1859 ) , retrieved 22 November 2008 Lucas , John
R.
</item 4>

<item 5>
On the Origin of Species
On the Origin of species - wikipedia On the Origin of species Jump to : navigation , search `` Origin of Species '' redirects here . For other uses , see Origin
of Species ( disambiguation ) . On the Origin of Species The title page of the 1859 edition of On the Origin of Species Author Charles Darwin Country United
Kingdom of Great Britain and Ireland Language English Subject Natural selection Evolutionary biology Published 24 November 1859 ( John Murray ) Media type Print
( Hardback ) Pages 502 OCLC 352242 Preceded by On the Tendency of Species to form Varieties ; and on the Perpetuation of Varieties and Species by Natural Means
of Selection Followed by Fertilisation of Orchids Part of a series on Evolutionary biology Key topics ( show ) Introduction to evolution Common descent Evidence
of common descent Processes and outcomes ( show ) Population genetics Variation Diversity Mutation Natural selection Adaptation Polymorphism Genetic drift Gene
flow Speciation Adaptive radiation Co-operation Coevolution Divergence Convergence Parallel evolution Extinction Natural history ( show ) Origin of life History
of life Timeline of evolution Human evolution Phylogeny Biodiversity Biogeography Classification Evolutionary taxonomy Cladistics Transitional fossil Extinction
event History
</item 5>

<item 6>
On the Origin of Species
Biodiversity Biogeography Classification Evolutionary taxonomy Cladistics Transitional fossil Extinction event History of evolutionary theory ( show ) Overview
Renaissance Before Darwin Darwin Origin of Species Before synthesis Modern synthesis Molecular evolution Evo - devo Current research History of speciation
History of paleontology ( timeline ) Fields and applications ( show ) Applications of evolution Biosocial criminology Ecological genetics Evolutionary
aesthetics Evolutionary anthropology Evolutionary computation Evolutionary ecology Evolutionary economics Evolutionary epistemology Evolutionary ethics
Evolutionary game theory Evolutionary linguistics Evolutionary medicine Evolutionary neuroscience Evolutionary physiology Evolutionary psychology Experimental
evolution Phylogenetics Paleontology Selective breeding Speciation experiments Sociobiology Systematics Universal Darwinism Social implications ( show )
Evolution as fact and theory Social effects Creation -- evolution controversy Objections to evolution Level of support Evolutionary biology portal Category Book
Related topics On the Origin of Species ( or more completely , On the Origin of Species by Means of Natural Selection , or the Preservation of Favoured Races in
the Struggle for Life ) , published on 24 November 1859 , is a work of scientific literature by Charles Darwin which is considered to be the foundation of
evolutionary biology .
</item 6>

<item 7>
On the Origin of Species
In a survey conducted by a group of academic booksellers , publishers and librarians in advance of Academic Book Week in the United Kingdom , On the Origin of
Species was voted the most influential academic book ever written . It was hailed as `` the supreme demonstration of why academic books matter '' and `` a book
which has changed the way we think about everything '' . See also ( edit ) Book : Evolution On the Origin of Species -- full text at Wikisource of the first
edition , 1859 The Origin of Species -- full text at Wikisource of the 6th edition , 1872 Charles Darwin bibliography History of biology History of evolutionary
thought History of speciation Modern evolutionary synthesis The Complete Works of Charles Darwin Online The Descent of Man , and Selection in Relation to Sex ,
published in 1871 ; his second major book on evolutionary theory . Transmutation of species Notes ( edit ) Jump up ^ Darwin 1859 , p. iii ^ Jump up to : Freeman
1977 ^ Jump up to : The book 's full original title was On the Origin of Species by Means of Natural Selection , or the Preservation of Favoured Races in the
Struggle for Life .
</item 7>

<item 8>
On the Origin of Species
Wikisource has original text related to this article : On the Origin of Species Library resources about On the origin of species Online books Resources in your
library Resources in other libraries The Complete Works of Charles Darwin Online : Table of contents , bibliography of On the Origin of Species -- links to text
and images of all six British editions of The Origin of Species , the 6th edition with additions and corrections ( final text ) , the first American edition ,
and translations into Danish , Dutch , French , German , Polish , Russian and Spanish . Online Variorum , showing every change between the six British editions
. On the Origin of Species on In Our Time at the BBC . On the Origin of Species , full text with embedded audio . Victorian Science Texts Darwin Correspondence
Project Home Page , University Library , Cambridge . On the Origin of Species eBook provided by Project Gutenberg View online at the Biodiversity Heritage
Library On the Origin of Species 1860 American edition , D Appleton and Company , New York , with front insert by H.E. Barker , Lincolniana .
</item 8>

<item 9>
On the Origin of Species
a Molecular Disease '' Fred Sanger Max Perutz John Kendrew Sydney Brenner Joshua Lederberg Walter Gilbert Kary Mullis Emmanuelle Charpentier Jennifer Doudna
Ecology Rachel Carson Frederic Clements Charles Elton Henry Gleason Arthur Tansley Eugenius Warming Ethology Niko Tinbergen Karl von Frisch Konrad Lorenz Frans
de Waal Jane Goodall Ivan Pavlov Related History of science Philosophy of biology Teleology Ethnobotany Eugenics History of the creation - evolution controversy
Human Genome Project Humboldtian science Natural history Natural philosophy Natural theology Relationship between religion and science Timeline of biology and
organic chemistry Category Commons Portal Evolutionary biology portal VIAF : 313683363 LCCN : n81105854 GND : 4303201 - 1 SELIBR : 261642 SUDOC : 031705685 BNF
: cb122870840 ( data ) NDL : 00627375 BNE : XX2039885 Retrieved from `` https://en.wikipedia.org/w/index.php?title=On_the_Origin_of_Species&oldid=836566097 ''
Categories : 1859 books 1859 in science Books about evolution
</item 9>

<item 10>
On the Origin of Species
December 1859 , pp. 188 -- 214 . Published anonymously . Gray , Asa ( 1860 ) , `` ( Review of ) The Origin of Species '' , Athenaeum ( 1710 : 4 August 1860 ) :
161 . Extract from Proceedings of the American Academy of Arts and Sciences 4 ( 1860 ) : 411 -- 415 . Huxley , Thomas Henry ( 1859 ) , `` Time and Life : Mr
Darwin 's Origin of Species '' , Macmillan 's Magazine , 1 : 142 -- 148 . Huxley , Thomas Henry ( 1859 ) , `` Darwin on the Origin of Species '' , The Times (
26 December 1859 ) : 8 -- 9 . Published anonymously . Jenkin , Fleeming ( 1867 ) , `` ( Review of ) The Origin of Species '' , North British Review , 46 ( June
1867 ) : 277 -- 318 . Published anonymously .
</item 10>
=== END ITEM id=fae7ea1a661c2693 ===

=== ITEM id=64bc05a0d95dc67f ===
Gold sentences and facts:
Sentence 0: The Third Five - year Plan , stressed agriculture and improvement in the production of wheat , but the brief Sino - Indian War of 1962 exposed weaknesses in the economy and shifted the focus towards the defence industry and the Indian Army . In 1965 -- 1966 , India fought a War with Pakistan . There was also a severe drought in 1965 . The war led to inflation and the priority was shifted to price stabilisation . The construction of dams continued . Many cement and fertilizer plants were also built . Punjab began producing an abundance of wheat .
  Fact 0.0: The Third Five-year Plan stressed agriculture.
  Fact 0.1: The Third Five-year Plan stressed improvement in the production of wheat.
  Fact 0.2: The brief Sino-Indian War of 1962 exposed weaknesses in the economy.
  Fact 0.3: The brief Sino-Indian War of 1962 shifted the focus towards the defence industry and the Indian Army.
  Fact 0.4: India fought a war with Pakistan in 1965-1966.
  Fact 0.5: There was a severe drought in 1965.
  Fact 0.6: The war led to inflation.
  Fact 0.7: The priority was shifted to price stabilisation.
  Fact 0.8: The construction of dams continued.
  Fact 0.9: Many cement plants were built.
  Fact 0.10: Many fertilizer plants were built.
  Fact 0.11: Punjab began producing an abundance of wheat.

Retrieved items:
<item 1>
Five-Year Plans of India
Hydroelectric power projects and five steel plants at Bhilai , Durgapur , and Rourkela were established with the help of Russia , Britain ( the U.K ) and West
Germany respectively . Coal production was increased . More railway lines were added in the north east . The Tata Institute of Fundamental Research and Atomic
Energy Commission of India was established as research institutes . In 1957 a talent search and scholarship program was begun to find talented young students to
train for work in nuclear power . The total amount allocated under the Second Five - Year Plan in India was Rs. 48 billion . This amount was allocated among
various sectors : power and irrigation , social services , communications and transport , and miscellaneous . `` The target growth rate was 4.5 % and the actual
growth rate was 4.27 % . '' Third Plan ( 1961 -- 1966 ) ( edit ) The Third Five - year Plan , stressed agriculture and improvement in the production of wheat ,
but the brief Sino - Indian War of 1962 exposed weaknesses in the economy and shifted the focus towards the defence industry and the Indian Army .
</item 1>

<item 2>
Five-Year Plans of India
The most important feature of this phase was active role of state in all economic sectors . Such a role was justified at that time because immediately after
independence , India was facing basic problems -- deficiency of capital and low capacity to save . The target growth rate was 2.1 % annual gross domestic
product ( GDP ) growth ; the achieved growth rate was 3.6 % the net domestic product went up by 15 % . The monsoon was good and there were relatively high crop
yields , boosting exchange reserves and the per capita income , which increased by 8 % . National income increased more than the per capita income due to rapid
population growth . Many irrigation projects were initiated during this period , including the Bhakra , Hirakud , Mettur Dam and Damodar Valley dams . The World
Health Organization ( WHO ) , with the Indian government , addressed children 's health and reduced infant mortality , indirectly contributing to population
growth . At the end of the plan period in 1956 , five Indian Institutes of Technology ( IITs ) were started as major technical institutions .
</item 2>

<item 3>
Five-Year Plans of India
Indian Economy : Problems of development and planning . pune : Wishwa Prakashan . p. 676 . Jump up ^ `` National Development Council approves 12th Five Year
Plan '' . Indian Express . 2012 - 12 - 27 . Retrieved 2013 - 07 - 10 . Jump up ^ ( 1 ) External links ( edit ) Official website of the Planning Commission of
India India topics History Overviews Timeline Years Astronomy Economics Linguistics Maritime Mathematics Metallurgy Military Postal Science and technology
Pre-colonial Stone Age Indus Valley Civilization Vedic period Mahajanapadas Mauryas Middle kingdoms Hoysala Chola Pala Kakatiya Delhi Sultanate Vijayanagara
Mughals Marathas European trade Colonial Princely East India Company Plassey 1857 rebellion British Raj Railways Economy Army Zamindari Bengali renaissance
Political reforms Princely states Partition of Bengal Independence movement 1943 famine World War II Partition Republic Integration Non-Aligned Movement Five -
Year Plans Sino - Indian War Indo - Pakistani wars Green Revolution White Revolution Naxal Insurgency Smiling Buddha Space programme The Emergency Indian Peace
Keeping Force ( IPKF ) Economic
</item 3>

<item 4>
Five-Year Plans of India
Contents ( hide ) 1 History 2 First Plan ( 1951 -- 1956 ) 3 Second Plan ( 1956 -- 1961 ) 4 Third Plan ( 1961 -- 1966 ) 5 Fourth Plan ( 1969 -- 1974 ) 6 Fifth
Plan ( 1974 -- 1978 ) 7 Rolling Plan ( 1978 -- 1980 ) 8 Sixth Plan ( 1980 -- 1985 ) 9 Seventh Plan ( 1985 -- 1990 ) 10 Annual Plans ( 1990 -- 1992 ) 11 Eighth
Plan ( 1992 -- 1997 ) 12 Ninth Plan ( 1997 -- 2002 ) 13 Tenth Plan ( 2002 -- 2007 ) 14 Eleventh Plan ( 2007 -- 2012 ) 15 Twelfth Plan ( 2012 -- 2017 ) 16 See
also 17 References 18 External links History ( edit ) Five - Year Plans ( FYPs ) are centralized and integrated national economic programs . Joseph Stalin
implemented the first Five Year Plan in the Soviet Union in the late 1920s .
</item 4>

<item 5>
Five-Year Plans of India
Most communist states and several capitalist countries subsequently have adopted them . China and India both continue to use FYPs , although China renamed its
Eleventh FYP , from 2006 to 2010 , a guideline ( guihua ) , rather than a plan ( jihua ) , to signify the central government 's more hands - off approach to
development . India launched its First FYP in 1951 , immediately after independence under socialist influence of first Prime Minister Jawaharlal Nehru . The
First Five - Year Plan was one of the most important because it had a great role in the launching of Indian development after Independence . Thus , it strongly
supported agriculture production and it also launched the industrialization of the country ( but less than the Second Plan , which focused on heavy industries )
. It built a particular system of mixed economy , with a great role for the public sector ( with an emerging welfare state ) , as well as a growing private
sector ( represented by some personalities as those who published the Bombay Plan ) . First Plan ( 1951 -- 1956 ) ( edit ) This article does not cite any
sources . Please help improve this article by adding citations to reliable sources .
</item 5>

<item 6>
Five-Year Plans of India
During 1966 -- 67 there was again the problem of drought . Equal priority was given to agriculture , its allied activities , and industrial sector . The
government of India declared `` Devaluation of Rupee '' to increase the exports of the country . The main reasons for plan holidays were the war , lack of
resources , and increase in inflation after that plan holiday was created . Fourth Plan ( 1969 -- 1974 ) ( edit ) At this time Indira Gandhi was the Prime
Minister . The Indira Gandhi government nationalised 14 major Indian banks and the Green Revolution in India advanced agriculture . In addition , the situation
in East Pakistan ( now Bangladesh ) was becoming dire as the Indo - Pakistan War of 1971 and Bangladesh Liberation War took funds earmarked for industrial
development . India also performed the Smiling Buddha underground nuclear test ( Pokhran - 1 ) in Rajasthan on May 18 , 1974 , partially in response to the
United States deployment of the Seventh Fleet in the Bay of Bengal . The fleet had been deployed to warn India against attacking West Pakistan and extending the
war .
</item 6>

<item 7>
Five-Year Plans of India
In 1965 -- 1966 , India fought a War with Pakistan . There was also a severe drought in 1965 . The war led to inflation and the priority was shifted to price
stabilisation . The construction of dams continued . Many cement and fertilizer plants were also built . Punjab began producing an abundance of wheat . Many
primary schools were started in rural areas . In an effort to bring democracy to the grass - root level , Panchayat elections were started and the states were
given more development responsibilities . State electricity boards and state secondary education boards were formed . States were made responsible for secondary
and higher education . State road transportation corporations were formed and local road building became a state responsibility . The target growth rate was 5.6
% , but the actual growth rate was 2.4 % . Due to miserable failure of the Third Plan the government was forced to declare `` plan holidays '' ( from 1966 -- 67
, 1967 -- 68 , and 1968 -- 69 ) . Three annual plans were drawn during this intervening period . During 1966 -- 67 there was again the problem of drought .
</item 7>

<item 8>
Five-Year Plans of India
The fleet had been deployed to warn India against attacking West Pakistan and extending the war . The target growth rate was 5.6 % , but the actual growth rate
was 3.3 % . Fifth Plan ( 1974 -- 1978 ) ( edit ) The Fifth Five - Year Plan laid stress on employment , poverty alleviation ( Garibi Hatao ) , and justice . The
plan also focused on self - reliance in agricultural production and defence . In 1978 the newly elected Morarji Desai government rejected the plan . The
Electricity Supply Act was amended in 1975 , which enabled the central government to enter into power generation and transmission . The Indian national highway
system was introduced and many roads were widened to accommodate the increasing traffic . Tourism also expanded . The twenty - point programme was launched in
1975 . It was followed from 1974 to 1979 . The Minimum Needs Programme ( MNP ) was introduced in the first year of the Fifth Five Year Plan ( 1974 -- 78 ) . The
objective of the programme is to provide certain basic minimum needs and thereby improve the living standards of the people .
</item 8>

<item 9>
Five-Year Plans of India
Five - Year plans of India - Wikipedia Five - Year plans of India Jump to : navigation , search From 1947 to 2017 , the Indian economy was premised on the
concept of planning . This was carried through the Five - Year Plans , developed , executed , and monitored by the Planning Commission ( 1951 - 2014 ) and the
NITI Aayog ( 2015 - 2017 ) . With the prime minister as the ex-officio chairman , the commission has a nominated deputy chairman , who holds the rank of a
cabinet Minister . Montek Singh Ahluwalia is the last deputy chairman of the commission ( resigned on 26 May 2014 ) . The Twelfth Plan completed its term in
March 2017 . Prior to the Fourth Plan , the allocation of state resources was based on schematic patterns rather than a transparent and objective mechanism ,
which led to the adoption of the Gadgil formula in 1969 . Revised versions of the formula have been used since then to determine the allocation of central
assistance for state plans .
</item 9>

<item 10>
Sino-Indian War
However , Indian patrols south of the McMahon line were expelled and suffered casualties in the Chinese attack . Zhou also told Nixon that Chairman Mao ordered
the troops to return to show good faith . The Indian government maintains that the Chinese military could not advance further south due to logistical problems
and the cut - off of resource supplies . While Western nations did not view Chinese actions favourably because of fear of the Chinese and competitiveness ,
Pakistan , which had had a turbulent relationship with India ever since the Indian partition , improved its relations with China after the war . Prior to the
war , Pakistan also shared a disputed boundary with China , and had proposed to India that the two countries adopt a common defence against `` northern ''
enemies ( i.e. China ) , which was rejected by India . However , China and Pakistan took steps to peacefully negotiate their shared boundaries , beginning on 13
October 1962 , and concluding in December of that year . Pakistan also expressed fear that the huge amounts of western military aid directed to India would
allow it to threaten Pakistan 's security in future conflicts .
</item 10>
=== END ITEM id=64bc05a0d95dc67f ===
