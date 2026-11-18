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

=== ITEM id=f3bdd4f15f83a7ed ===
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
High School Musical 2
Taylor McKessie ( Monique Coleman ) is Gabriella 's best friend . She is also friends with Kelsi Nielsen and Martha Cox and is dating Chad . She is captain of
the school Scholastic Decathlon team . She has a summer job at Lava Springs as an Activities Coordinator . Thomas Fulton ( Mark L. Taylor ) is the manager of
Lava Springs . Coach Jack Bolton ( Bart Johnson ) is Troy 's father and East High 's basketball coach . Darby Evans ( Jessica Tuck ) is Sharpay and Ryan 's
mother and president of the Lava Springs Board . Vance Evans ( Robert Curtis Brown ) is Sharpay and Ryan 's father . Ms. Darbus ( Alyson Reed ) is the stern
drama teacher at East High . Zeke Baylor ( Chris Warren Jr . ) is friends with Troy and Chad and plays for the basketball team . Jason Cross ( Ryne Sanborn ) is
friends with Troy , Chad , and Zeke and plays on the basketball team . Kelsi Nielsen ( Olesya Rulin ) is friends with Gabriella and Taylor and a pianist and
composer .
</item 1>

<item 2>
Will Friedle
Will Friedle - wikipedia Will Friedle Jump to : navigation , search Will Friedle William Alan Friedle ( 1976 - 08 - 11 ) August 11 , 1976 ( age 41 ) Hartford ,
Connecticut , U.S. Occupation Actor , voice actor Years active 1988 -- present Spouse ( s ) Susan Martens ( m . 2016 ) William Alan `` Will '' Friedle ( born
August 11 , 1976 ) is an American actor and voice actor . He is best known for his comedic roles , most notably the underachieving elder brother Eric Matthews
on the long - running TV sitcom Boy Meets World from 1993 to 2000 . More recently , he has voiced a number of animated characters such as Terry McGinnis /
Batman , the title character of Batman Beyond , and Ron Stoppable of Kim Possible . He voices Deadpool in Ultimate Spider - Man and Star - Lord in the animated
version of Guardians of the Galaxy , replacing Chris Cox . He also performed the voices of Doyle in The Secret Saturdays , Lion - O in the rebooted ThunderCats
series , and Blue Beetle on Batman : The Brave and the Bold .
</item 2>

<item 3>
Will Friedle
Career ( edit ) After co-hosting Nickelodeon 's Do n't Just Sit There for three seasons ( 1988 -- 1991 ) , Friedle was cast as Cory Matthews ' likeable but
underachieving elder brother Eric on the long - running TV sitcom Boy Meets World from 1993 to 2000 . In 2004 , he co-starred with Chris Owen in the independent
film National Lampoon 's Gold Diggers ( originally titled Lady Killers ) . He also appeared in a supporting role in the 2005 ABC Family movie Everything You
Want . Friedle was cast as Mike in the WB sitcom Off Centre , but could not accept the role because of a previous commitment to the UPN series The Random Years
. The role ended up going to Eddie Kaye Thomas . Friedle voiced Seifer Almasy in the English version of the video game Kingdom Hearts II , Gideon Wyeth in the
English version of the video game Advent Rising , and Jaster Rogue in the English version of the video game Rogue Galaxy .
</item 3>

<item 4>
Will Friedle
Personal life ( edit ) In 1997 , Friedle dated American actress Jennifer Love Hewitt , and co-starred with her in the romantic teen comedy Trojan War . Hewitt
made a guest appearance as Jennifer Love `` Feffy '' Fefferman on Boy Meets World episode `` And Then There Was Shawn '' . Friedle is best friends with actor
Jason Marsden and was the best man at Marsden 's wedding in October 2004 . The two have also worked together in numerous productions : Boy Meets World , Trojan
War , Static Shock , Kim Possible , Batman Beyond , Justice League Unlimited , Batman : The Brave and the Bold , ThunderCats and Mad . He married his girlfriend
, Susan Martens , on September 25 , 2016 in Connecticut . Filmography ( edit ) Film Year Film Role Notes Trojan War Brad Kimble 2000 Batman Beyond : Return of
the Joker Terry McGinnis / Batman Voice role Direct - to - DVD release 2003 National Lampoon 's Gold Diggers Calvin Menhoffer The Greatest Short Film Ever ! ! !
</item 4>

<item 5>
Will Friedle
! ! Mark Short film Howl 's Moving Castle Additional voices Voice role 2005 Everything You Want Calvin Dillwaller 2008 Batman : Gotham Knight Anton , Cultist ,
Youth # 1 Voice role Direct - to - DVD release 2015 Batman Unlimited : Animal Instincts Nightwing / Dick Grayson Voice role Direct - to - DVD release Batman
Unlimited : Monster Mayhem Nightwing / Dick Grayson Voice role Direct - to - DVD release 2016 Lego DC Comics Super Heroes : Justice League : Gotham City
Breakout Nightwing / Dick Grayson Voice role Direct - to - DVD release Batman Unlimited : Mech vs. Mutants Nightwing / Dick Grayson Voice role Direct - to - DVD
release Television Year Title Role Notes Do n't Just Sit There Host Law & Order Russ Episode : `` Trust '' 1993 -- 2000 Boy Meets World Eric Matthews Series
regular ( 157 episodes ) Young Artist Award ( nominated ) The Gift of Love Luke CBS television movie Are You Afraid of the Dark ?
</item 5>

<item 6>
Daren Kagasoff
Daren Kagasoff - wikipedia Daren Kagasoff Jump to : navigation , search Daren Kagasoff At the Twilight premiere Daren Maxwell Kagasoff ( 1987 - 09 - 16 )
September 16 , 1987 ( age 30 ) Encino , California , U.S. Occupation Actor Years active 2008 -- present Daren Maxwell Kagasoff ( KA - guh - sawf ) ( born
September 16 , 1987 ) is an American actor . He is known for starring as Ricky Underwood on the ABC Family teen drama series The Secret Life of the American
Teenager from 2008 to 2013 . Contents ( hide ) 1 Early life 2 Career 3 Filmography 4 Awards and nominations 5 References 6 External links Early Life ( edit )
Kagasoff was born in Encino , California to Elise Kagasoff ( née Levy ) and Barry Kagasoff , a diamond merchant . He is the middle of three children ; he has an
older brother , Justin , and a younger sister , Natalie .
</item 6>

<item 7>
Will Friedle
Kiel ( 28 January 2009 ) . `` The Bravest & Boldest : Will Friedle '' . Comic Book Resources . Retrieved 30 November 2010 . Jump up ^ Andy Swift . `` ' Girl
Meets World ' : Eric Returns With Mr. Feeny -- Season 2 Spoilers - TVLine '' . TVLine . Retrieved 4 April 2016 . Jump up ^ `` Girl Meets Writers '' . Twitter .
Retrieved 4 April 2016 . Jump up ^ `` Will Friedle on Twitter '' . Twitter . Retrieved 4 April 2016 . Jump up ^ `` Art - World Movie Shot In State To Be Shown
Saturday In New Milford '' . tribunedigital - thecourant . Retrieved 4 April 2016 . Jump up ^ `` The Hartford Courant article archive - Friedle - Leary '' .
Retrieved 4 April 2016 . Jump up ^ `` The Hartford Courant article archive - She Chose to Have Both '' . Retrieved 4 April 2016 .
</item 7>

<item 8>
Will Friedle
One of his most recent roles has been the speaking voice of Bumblebee in the final episode of Transformers : Prime and the movie Transformers Prime Beast
Hunters : Predacons Rising and in the sequel series Transformers : Robots in Disguise , as well as in Transformers : Rescue Bots . In December 2013 , it was
announced that Friedle would reprise his Boy Meets World role of Eric Matthews on the continuation series Girl Meets World . Contents ( hide ) 1 Early life 2
Career 3 Personal life 4 Filmography 5 References 6 External links Early life ( edit ) Friedle was born in Hartford , Connecticut , the son of Patricia Joan (
née Leary ) and Gary Allen Friedle , both lawyers . After committing himself to becoming an actor , Friedle continually commuted from Avon , Connecticut to New
York City for auditions . He attended Avon High School in 1994 . Friedle attended Occidental College . In the July 4 , 2006 issue of People , Friedle mentioned
he had thought about attending culinary school .
</item 8>

<item 9>
Will Friedle
Retrieved 4 April 2016 . Jump up ^ Friedle lands UPN sitcom Neil Wilkes , DigitalSpy , 4 March 2002 Jump up ^ `` Will Friedle '' . Retrieved 4 April 2016 . Jump
up ^ `` Boy Meets World star Will Friedle celebrates his marriage '' . Retrieved 15 August 2017 . Jump up ^ `` Gary Miereanu '' . Twitter . Retrieved 4 April
2016 . Jump up ^ Gerding , Stephen ( 2 June 2015 ) . `` EXCLUSIVE TRAILER DEBUT : Joker Rules in `` Batman Unlimited : Monster Mayhem '' `` . Comic Book
Resources . Retrieved 6 June 2015 . Jump up ^ Siegel , Lucas ( April 4 , 2016 ) . `` EXCLUSIVE : WB Reveals Next LEGO DC Comics Movie with Announcement Trailer
'' . Comic Book Resources . Retrieved July 25 , 2016 . Jump up ^ `` New Frontiers '' . Avengers Assemble . Season 2 . Episode 25 . September 13 , 2015 .
</item 9>

<item 10>
Daren Kagasoff
The film was released in October 2014 . In September 2014 , Kagasoff joined the cast of the Red Band Society , a comedy - drama series on Fox for the 2014 -- 15
American television season developed by Margaret Nagle . He had a recurring role playing Kara 's love interest , Hunter Cole . The series premiered on September
17 , 2014 and was canceled after one season . Filmography ( edit ) Television and film roles Year Title Role Notes 2008 -- 2013 Secret Life of the American
Teenager , The The Secret Life of the American Teenager Richard `` Ricky '' Underwood Main role 2014 Ouija Trevor Film 2014 Blue Daren Web series ; 2 episodes
2014 Delirium Alex Sheathes Unsold television pilot ( FOX ) 2014 Stalker Eric Bates Episode : Pilot 2014 -- 2015 Red Band Society Hunter Cole Recurring role , 7
episodes 2015 Paradise Pictures Wyatt Osborne Unsold television pilot ( USA ) 2017 S.W.A.T.
</item 10>
=== END ITEM id=f3bdd4f15f83a7ed ===

=== ITEM id=d1bb0dafbfd56259 ===
Gold sentences and facts:
Sentence 0: From September 1947 , Toyota 's small - sized vehicles were sold under the name `` Toyopet '' ( トヨペット ) . The first vehicle sold under this name was the Toyopet SA , but it also included vehicles such as the Toyopet SB light truck , Toyopet Stout light truck , Toyopet Crown , Toyopet Master , and the Toyopet Corona . The word `` Toyopet ( Japanese article ) '' was a nickname given to the Toyota SA due to its small size , as the result of a naming contest the Toyota Company organized in 1947 . However , when Toyota eventually entered the American market in 1957 with the Crown , the name was not well received due to connotations of toys and pets . The name was soon dropped for the American market , but continued in other markets until the mid-1960s .
  Fact 0.0: From September 1947, Toyota's small-sized vehicles were sold under the name 'Toyopet' (トヨペット).
  Fact 0.1: The first vehicle sold under the name Toyopet was the Toyopet SA.
  Fact 0.2: Toyopet vehicles also included the Toyopet SB light truck.
  Fact 0.3: Toyopet vehicles also included the Toyopet Stout light truck.
  Fact 0.4: Toyopet vehicles also included the Toyopet Crown.
  Fact 0.5: Toyopet vehicles also included the Toyopet Master.
  Fact 0.6: Toyopet vehicles also included the Toyopet Corona.
  Fact 0.7: The word 'Toyopet' was a nickname given to the Toyota SA due to its small size.
  Fact 0.8: The Toyopet nickname resulted from a naming contest the Toyota Company organized in 1947.
  Fact 0.9: When Toyota entered the American market in 1957 with the Crown, the Toyopet name was not well received due to connotations of toys and pets.
  Fact 0.10: The Toyopet name was soon dropped for the American market.
  Fact 0.11: The Toyopet name continued in other markets until the mid-1960s.

Retrieved items:
<item 1>
Toyota
The principle of jidoka , which means the machine stops itself when a problem occurs , became later a part of the Toyota Production System . Looms were built on
a small production line . In 1929 , the patent for the automatic loom was sold to the British company Platt Brothers , generating the starting capital for the
automobile development . Mass production of Toyoda automated loom , displayed at the Toyota Museum in Nagakute - cho , Aichi - gun , Aichi Pref . Japan The
production of Toyota automobiles was started in 1933 as a division of Toyoda Automatic Loom Works devoted to the production of automobiles under the direction
of the founder 's son , Kiichiro Toyoda . Its first vehicles were the A1 passenger car and the G1 in 1935 . The Toyota Motor Co. was established as an
independent company in 1937 Toyoda Standard Sedan AA 1936 Vehicles were originally sold under the name `` Toyoda '' ( トヨダ ) , from the family name of the
company 's founder , Kiichirō Toyoda . In April 1936 , Toyoda 's first passenger car , the Model AA , was completed .
</item 1>

<item 2>
Toyota, founded in 1937 from Toyoda loom works, grew into a global automaker with hundreds of subsidiaries, extensive model and dealership networks, and brands
including Lexus. Its Toyota Production System emphasizes just-in-time manufacturing, quality, teamwork, and continuous improvement. By 2017, Toyota and Lexus
had sold 10 million hybrids; the Prius family exceeded six million, with major fuel and CO₂ savings. The company also operates in finance, robotics, motorsport,
marine and other
</item 2>

<item 3>
Toyota
Toyota Technical Center , Ann Arbor Charter Township , Michigan Toyota has a large presence in the United States with six major assembly plants in Huntsville ,
Alabama , Georgetown , Kentucky , Princeton , Indiana , San Antonio , Texas , Buffalo , West Virginia , and Blue Springs , Mississippi . Toyota had a joint -
venture operation with General Motors at New United Motor Manufacturing Inc. in Fremont , California , which began in 1984 and ended in 2009 . It still has a
joint venture with Subaru at Subaru of Indiana Automotive , Inc. in Lafayette , Indiana , which started in 2006 . In these assembly plants , the Camry and the
Tundra are manufactured , among others . Toyota marketing , sales , and distribution in the US are conducted through a separate subsidiary , Toyota Motor Sales
, U.S.A. , Inc . It has started producing larger trucks , such as the new Tundra , to go after the large truck market in the United States . Toyota is also
pushing hybrid vehicles in the US such as the Prius , Camry Hybrid , Highlander Hybrid , and various Lexus products . Currently , Toyota has no plans to offer
diesel motor options in its North American products , including the light - duty pickup trucks .
</item 3>

<item 4>
Toyota
Akio Toyoda became the new president and CEO of the company on June 23 , 2009 , by replacing Katsuaki Watanabe , who became the new vice chairman by replacing
Katsuhiro Nakagawa . Worldwide presence ( edit ) The Toyota Camry is assembled in several facilities around the world including Australia , China , Taiwan ,
Japan , Malaysia , the Philippines , Pakistan , Russia , Thailand , India , Vietnam , and the United States Toyota has factories in most parts of the world ,
manufacturing or assembling vehicles for local markets in Japan , Australia , India , Sri Lanka , Canada , Indonesia , Poland , South Africa , Turkey , Colombia
, the United Kingdom , the United States , France , Brazil , Portugal , and more recently , Argentina , Czech Republic , Mexico , Malaysia , Thailand , Pakistan
, Egypt , China , Vietnam , Venezuela , the Philippines , and Russia .
</item 4>

<item 5>
Toyota
However , when Toyota eventually entered the American market in 1957 with the Crown , the name was not well received due to connotations of toys and pets . The
name was soon dropped for the American market , but continued in other markets until the mid-1960s . 1960s -- 1970s ( edit ) By the early 1960s , the US had
begun placing stiff import tariffs on certain vehicles . The so - called `` chicken tax '' of 1964 placed a 25 % tax on imported light trucks . In response to
the tariff , Toyota , Nissan Motor Co. and Honda Motor Co. began building plants in the US by the early 1980s . 1980s ( edit ) With over 40 million sold , the
Corolla is one of the most popular and best selling cars in the world . Toyota received its first Japanese Quality Control Award at the start of the 1980s and
began participating in a wide variety of motorsports . Due to the 1973 oil crisis , consumers in the lucrative US market began turning to small cars with better
fuel economy .
</item 5>

<item 6>
More than 7 million U.S. vehicles may be affected by a recall involving Takata airbags manufactured after 2000 for North America; Toyota offers free repairs
worldwide, and other North American automakers were also affected. Toyota entered the passenger-car market in 1936 with the Model AA and chose a new logo from
27,000 entries, changing its name from Toyoda to Toyota. The eight-stroke Japanese spelling was associated with good fortune. Its original logo remains a
corporate emblem in
</item 6>

<item 7>
Toyota Motor Company was registered in 1937. Its postwar “Toyopet” name covered models including the Crown and Corona, but was dropped in the U.S. because it
suggested toys and pets. After the 1973 oil crisis boosted demand for fuel-efficient cars, Toyota grew the Corolla’s global success, won a Japanese quality
award, and entered motorsports. It merged with Toyota Motor Sales in 1982, partnered with GM at NUMMI, and launched Lexus in 198
</item 7>

<item 8>
In early 2007, Toyota and Daihatsu sold 2.348 million vehicles; Toyota sales rose 9.2%, driven by Corolla and Camry demand. Toyota held over 16% of the U.S.
market and ranked second by volume. It opened a San Antonio facility in 2006 but faced quality and recall-practice criticism. The 2008 crisis led to its first
projected annual loss in 70 years and an 11-day Japanese plant shutdown. The Century
</item 8>

<item 9>
Toyota is a Japanese multinational manufacturer founded in 1937, known for cars and hybrids and also involved in electric vehicles, marine engines, military
vehicles, defense, and robotics. The text lists Toyota racing cars, including the TS010, TS020, and TS030–TS050 hybrids, alongside Japanese automakers, parts
suppliers, tuning firms, and bus manufacturers. It also places Toyota among Japan’s major Nikkei 225 and TOPIX companies.
</item 9>

<item 10>
Toyota began as an automobile division of Toyoda Automatic Loom Works in 1933, led by founder Kiichiro Toyoda. It developed the Type A engine in 1934 and its
first vehicles, the A1 and G1, in 1935; the Model AA passenger car followed in 1936. Toyota Motor Corporation became independent in 1937. Loom-patent sales to
Platt Brothers helped fund automobile development, while the jidoka principle later shaped Toyota’s production
</item 10>
=== END ITEM id=d1bb0dafbfd56259 ===

=== ITEM id=4ad68157ea4a2d96 ===
Gold sentences and facts:
Sentence 0: A cover of the song by Stacey Markus is used at the end of episode nine ( `` The Invitation '' ) of the 2015 Netflix original series Grace and Frankie , starring Jane Fonda and Lily Tomlin .
  Fact 0.0: A cover of a song by Stacey Markus is used at the end of episode nine, 'The Invitation', of Grace and Frankie.
  Fact 0.1: Grace and Frankie is a 2015 Netflix original series.
  Fact 0.2: Grace and Frankie stars Jane Fonda.
  Fact 0.3: Grace and Frankie stars Lily Tomlin.

Retrieved items:
<item 1>
Let My Love Open the Door
A cover of the song by Stacey Markus is used at the end of episode nine ( `` The Invitation '' ) of the 2015 Netflix original series Grace and Frankie ,
starring Jane Fonda and Lily Tomlin . The song is featured in a Walmart Christmas holiday television commercial which started airing in early December 2017 .
Lethal Weapon ( TV series ) , season 2 , episode 9 `` Fools Rush In '' play the song performed by `` Rogue wave '' ( 2017 ) . References ( edit ) Jump up ^
Unterberger , Richie ( 2012 - 06 - 14 ) . `` Let My Love Open the Door '' . Rolling Stone . Retrieved 2016 - 10 - 16 . Jump up ^ `` Pete Townshend , ' Empty
Glass ' 100 Best Albums of the Eighties '' . Rolling Stone . 1989 - 11 - 16 . Retrieved 2016 - 10 - 16 . ^ Jump up to : Bac-lac.gc.ca ^ Jump up to : `` Let My
Love Open the Door '' . Songfacts.com .
</item 1>

<item 2>
Let My Love Open the Door
It reached number five in Canada . Contents ( hide ) 1 Background 2 Chart performance 2.1 Weekly charts 2.2 Year - end charts 3 Personnel 4 Covers 5 List of
Films , TV Shows , and Other Media featuring the track 6 References 7 External links Background ( edit ) Although Townshend is a devotee to the religious guru
Meher Baba , he claimed in the liner notes of his Anthology CD that `` Jesus sings '' on the track . `` Let My Love Open the Door '' was released as the second
single from Empty Glass in Britain , where it was backed with the non-album tracks `` Classified '' and `` Greyhound Girl . '' The song was a minor British hit
, reaching # 46 . The song saw more success when it was released as the debut single from Empty Glass in America , where the song reached # 9 . It was Pete
Townshend 's only solo top 10 hit on the Billboard Hot 100 , but The Who 's song `` I Can See For Miles , '' which was written by Townshend , reached the same
position on the chart 13 years earlier .
</item 2>

<item 3>
Let My Love Open the Door
Imdb.com. 2017 - 05 - 12 . Retrieved 2018 - 03 - 30 . External links ( edit ) Lyrics of this song at MetroLyrics ( hide ) Pete Townshend Discography Studio
albums Who Came First Rough Mix ( w / Ronnie Lane ) Empty Glass All the Best Cowboys Have Chinese Eyes White City : A Novel The Iron Man : A Musical
Psychoderelict Classic Quadrophenia Meher Baba tribute albums ( w / Ronnie Lane et al . ) Happy Birthday I Am With Love O ' Parvardigar Live albums Deep End
Live !
</item 3>

<item 4>
Let My Love Open the Door
Let My Love Open the Door - wikipedia Let My Love Open the Door Jump to : navigation , search `` Let My Love Open the Door '' Single by Pete Townshend from the
album Empty Glass B - side `` Greyhound Girl '' Released June 4 , 1980 ( US ) Format 7 '' Recorded March 1980 Genre Art rock , new wave Length 2 : 44 Label Atco
Records Songwriter ( s ) Pete Townshend Producer ( s ) Pete Townshend ,
</item 4>

<item 5>
Let My Love Open the Door
The group Rogue Wave also covered it in `` Cover me '' released in 2017 . List of films , TV shows , and other media featuring the track ( edit ) The song has
been used frequently in film , most notably the comedy genre , often as trailer music for movies such as Jerry Maguire and How Do You Know , and then as end
credit song for Mr. Deeds in 2002 ; Jersey Girl and Along Came Polly , both in 2004 . It was featured in the following films and TV series : Look Who 's Talking
( 1989 ) Grosse Pointe Blank as `` E. Cola Mix '' ( 1997 ) Mr.
</item 5>

<item 6>
Let My Love Open the Door
Initially , Townshend 's manager despised the track due to it `` not sounding like Townshend , '' and wanted it to be removed from Empty Glass . However , upon
the song 's chart success , his manager called to apologize . Despite the song 's critical and commercial success , Pete Townshend did not consider it one of
his best songs . He told Rolling Stone in an interview that `` Let My Love Open the Door '' was `` just a ditty , '' also claiming that he preferred his minor
US hit `` A Little Is Enough '' from the same album . In 1996 , Townshend released a new version of the song , called `` the E. Cola mix '' , turning the song
into a ballad . This version appeared in different television shows and film soundtracks . Chart performance ( edit ) Weekly charts ( edit ) Chart ( 1980 ) Peak
position Canada RPM 100 5 UK Singles Chart 46 U.S. Billboard Hot 100 9 U.S. Cash Box Top 100 11 Year - end charts ( edit ) Chart ( 1980 ) Rank Canada 35 U.S.
</item 6>

<item 7>
Let My Love Open the Door
Chris Thomas Pete Townshend UK singles chronology `` Rough Boys '' ( 1980 ) `` Let My Love Open the Door '' ( 1980 ) `` Keep on Working '' ( 1980 ) `` Rough
Boys '' ( 1980 ) `` Let My Love Open the Door '' ( 1980 ) `` Keep on Working '' ( 1980 ) Pete Townshend US singles chronology `` Keep Me Turning '' ( 1977 ) (
with Ronnie Lane ) Keep Me Turning1977 `` Let My Love Open the Door '' ( 1980 ) Let My Love Open the Door1980 `` A Little Is Enough '' ( 1980 ) A Little Is
Enough 1980 `` Let My Love Open the Door '' is a song written and performed by Pete Townshend from his 1980 album Empty Glass . It reached the top ten in the
United States in that same year , reaching number nine . It reached number five in Canada .
</item 7>

<item 8>
Let My Love Open the Door
at 22 : 30 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the
Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 8>

<item 9>
Let My Love Open the Door
Billboard Hot 100 59 U.S. Cashbox 92 Personnel ( edit ) Pete Townshend : vocals , guitars , synths John `` Rabbit '' Bundrick : `` straight '' keyboards Simon
Phillips : drums Tony Butler : bass guitar Ted Jensen - mastering engineer Covers ( edit ) A version of this song by Luminate appears on The Covers EP . It was
also covered by M. Ward ( for the 2005 Starbucks compilation album , Sweetheart 2005 : Love Songs ) Christian rock group Audio Adrenaline covered this song on
their 1999 album Underdog . Natalie Imbruglia covered the song on her album Male released in 2015 . Great Big Sea covered the song on their 2012 album XX .
Roger Daltrey covered the song in 2015 for a campaign involving Teen Cancer America and First Citizens Bank . A full version was released in February 2016 ,
with all proceeds going to Teen Cancer America . The group Rogue Wave also covered it in `` Cover me '' released in 2017 .
</item 9>

<item 10>
“The Glory of Love,” written by Billy Hill, was a 1936 Benny Goodman pop hit. The Five Keys’ 1951 version topped the R&B chart for four nonconsecutive weeks;
Otis Redding’s 1967 cover reached No. 19 R&B and No. 60 on the Hot 100.

Pete Townshend wrote and recorded “Let My Love Open the Door” for his 1980 album *Empty Glass*. It peaked at No. 9
</item 10>
=== END ITEM id=4ad68157ea4a2d96 ===

=== ITEM id=836b3525c2549248 ===
Gold sentences and facts:
Sentence 0: On May 1 , 2017 , Slasher began filming a second season in Orangeville , Ontario , Canada . Shaftesbury Films confirmed that the series will not be returning to Super Channel or Chiller for a second season . However , Netflix acquired the rights to the second season . The second season was released via Netflix on October 17 , 2017 .
  Fact 0.0: Slasher began filming a second season on May 1, 2017.
  Fact 0.1: Slasher filmed its second season in Orangeville, Ontario, Canada.
  Fact 0.2: Shaftesbury Films confirmed that Slasher will not return to Super Channel or Chiller for a second season.
  Fact 0.3: Netflix acquired the rights to the second season of Slasher.
  Fact 0.4: The second season of Slasher was released via Netflix on October 17, 2017.
Sentence 1: The first season , retroactively subtitled The Executioner , was produced in association with the Canadian network Super Channel , Slasher was the first original series by U.S. TV channel Chiller , which premiered the series on Friday , March 4 , 2016 , at 9 : 00 pm EST . Super Channel aired the show 's Canadian premiere on April 1 , 2016 . The series first season , which centered on a mysterious figure billed as `` The Executioner '' terrorizing the fictional town of Waterbury , Canada , was filmed between July and October 2015 in Sudbury , Parry Sound , and Sault Ste . Marie , Ontario .
  Fact 1.0: The first season of Slasher, retroactively subtitled The Executioner, was produced in association with the Canadian network Super Channel.
  Fact 1.1: Slasher was the first original series by U.S. TV channel Chiller.
  Fact 1.2: Chiller premiered the series on Friday, March 4, 2016, at 9:00 pm EST.
  Fact 1.3: Super Channel aired the show's Canadian premiere on April 1, 2016.
  Fact 1.4: The first season of Slasher centered on a mysterious figure billed as 'The Executioner'.
  Fact 1.5: The Executioner terrorized the fictional town of Waterbury, Canada.
  Fact 1.6: The first season was filmed between July and October 2015.
  Fact 1.7: The first season was filmed in Sudbury, Parry Sound, and Sault Ste. Marie, Ontario.

Retrieved items:
<item 1>
Slasher (TV series)
of seasons No. of episodes 16 ( list of episodes ) Production Executive producer ( s ) Aaron Martin Greg Phillips Jonathan Ford Christina Jennings Scott Garvie
Location ( s ) Ontario , Canada Editor ( s ) Erin Deck Camera setup Single - camera Running time 46 - 53 minutes Production company ( s ) Shaftesbury Films
Super Channel ( season 1 ) TVA Group ( season 2 ) Distributor Content Media Corporation Release Original network Super Channel ( Canada , season 1 ) Chiller (
USA , season 1 ) Netflix ( season 2 ) Original release March 4 , 2016 ( 2016 - 03 - 04 ) -- present External links Website www.netflix.com/title/80105452
Slasher is a Canadian television horror anthology series created by Aaron Martin . The first season , retroactively subtitled The Executioner , was produced in
association with the Canadian network Super Channel , Slasher was the first original series by U.S. TV channel Chiller , which premiered the series on Friday ,
March 4 , 2016 , at 9 : 00 pm EST . Super Channel aired the show 's Canadian premiere on April 1 , 2016 .
</item 1>

<item 2>
Slasher (TV series)
However , Netflix acquired the rights to the second season . The second season was released via Netflix on October 17 , 2017 . Reception ( edit ) Slasher has
received positive reviews . Zap2it called the series `` a whole lot of fun '' and `` something for everyone , '' praising the series ' anthology nature , its
cast of characters , storyline , plot twists , bloody violence , and even the series - within - the - series Falcon Husbandry ( shown as a favorite of Robin and
Justin 's in Episode 2 ) . Bloody Disgusting awarded the show four skulls out of five , praising Katie McGrath as a great `` protagonist and possible final girl
'' and the series ' decision to feature an adult cast , rather than teenagers , with well - developed characters and a `` decidedly classic '' presentation . On
the occasion of Slasher 's Super Channel premiere , The Globe and Mail 's John Doyle , while remarking that it `` is no masterpiece of horror , nor was it meant
to be , '' called the show `` very well - crafted , '' praising its `` exceptional cast '' and tight pacing , and noting fans of gory horror will appreciate its
bloody scenes .
</item 2>

<item 3>
Slasher (TV series)
Not wanting to have the show 's killer be `` a mythological creature '' ( as he feels the killers in most slasher films do not have much mystery surrounding
them ) , Martin also uses elements of the traditional whodunit in Slasher : The show 's characters , many of whom have mysterious backgrounds -- and their own
reasons for possibly being the killer -- are featured , explored , and eliminated from consideration , one by one ( either through death or the natural
deductive process ) , until the `` all too human '' killer and their motivations for their actions are revealed . Plot ( edit ) Season 1 : the Executioner (
2016 ) ( edit ) Slasher 's 8 - episode first season centers around Sarah Bennett who , with husband Dylan , moves back to the town of her birth , Waterbury ( a
fictional town in Canada ) , and into her parents ' former home . It was in that house on Halloween Night 1988 where both of her parents , Bryan and Rachel ,
were murdered . Rachel was pregnant with Sarah at the time of her murder , with police discovering the killer holding Rachel 's newborn baby after the slayings
.
</item 3>

<item 4>
Slasher (TV series)
The Futon Critic . July 28 , 2016 . Retrieved March 6 , 2016 . ^ Jump up to : `` Lights , camera , action ... ' Slasher ' TV series begins filming in
Orangeville '' . Orangeville Banner . May 1 , 2017 . Retrieved May 2 , 2017 . ^ Jump up to : `` Slasher returns for a second season '' . Playback Online . May 9
, 2017 . Retrieved September 29 , 2017 . Jump up ^ Moore , Kasey ( September 20 , 2017 ) . `` October 2017 New Netflix Releases '' . WhatsonNetlix . Retrieved
September 20 , 2017 . Jump up ^ Urqyhart - White , Alaina ( October 17 , 2017 ) . `` Is ' Slasher 2 : Guilty Party ' A True Story ? The Netflix Series Honors
Some Legendary Horror Classics '' . Bustle . Retrieved October 18 , 2017 . ^ Jump up to : Patrick Hipes .
</item 4>

<item 5>
Slasher (TV series)
Season 2 : Guilty Party ( 2017 ) ( edit ) Slasher 's 8 - episode second season centers around a horrific secret that a group of former summer camp counselors
have buried long ago . Due to a changing circumstance , they must return to their resort in the dead of winter to retrieve the body of Talvinder , whom they
murdered while they were camp counsellors , five years earlier . They get surprised when they ca n't find the body at the campsite . Later , the group start
dying one by one at the hands of a mysterious person wearing a parka , which also alerts the second group of counselors at the resort . Eventually , it 's
revealed that Judith is the killer , and another character , Wren , is actually Owen `` Wren '' Turnbull , her son who committed suicide after being framed for
Talvinder 's murder , and is a figment of Judith 's imagination . Owen wrote a letter to his mother , asking her to get justice for him before hanging himself .
</item 5>

<item 6>
Slasher (TV series)
Slasher ( TV series ) - wikipedia Slasher ( TV series ) Jump to : navigation , search Slasher Genre Anthology Horror Slasher Drama Created by Aaron Martin
Starring Katie McGrath Brandon Jay McLaren Steve Byers Patrick Garrow Dean McDermott Leslie Hope Lovell Adams - Gray Jim Watson Christopher Jacot Paulino Nunes
Ty Olsson Joanne Vannicola Sebastian Pigott Madison Cheeatow Melinda Shankar Kaitlyn Leeb Rebecca Liddiard Paula Brancati Composer ( s ) Shawn Pierce Country of
origin Canada Original language ( s ) English No. of seasons No.
</item 6>

<item 7>
Slasher (TV series)
A first trailer for the series would be released on November 26 , 2015 . Unlike most TV series that film their episodes in order , Slasher , under the direction
of Craig David Wallace , was shot as if it were a `` super-sized '' movie : Scenes from multiple episodes were shot at the same time , with the availability of
locations and cast being factored in . The out - of - order schedule allowed the actors to know of their characters ' fates , especially those who had to film
their death scenes one day but return later to film earlier scenes as necessary . As an example of this , Martin cited Mark Ghanimé 's first day on set , when
his character , Justin Faysal , was laid out in a casket for a scene early in Slasher 's third episode ( Justin 's death , which took place in Episode 2 , would
be filmed later on ) . On May 1 , 2017 , Slasher began filming a second season in Orangeville , Ontario , Canada . Shaftesbury Films confirmed that the series
will not be returning to Super Channel or Chiller for a second season .
</item 7>

<item 8>
Slasher (TV series)
Contents ( hide ) 1 Background 2 Plot 2.1 Season 1 : The Executioner ( 2016 ) 2.2 Season 2 : Guilty Party ( 2017 ) 3 Cast and characters 3.1 Recurring cast
members 3.2 Season 1 : The Executioner 3.2. 1 Main 3.2. 2 Recurring 3.3 Season 2 : Guilty Party 4 Episodes 4.1 Season 1 : The Executioner ( 2016 ) 4.2 Season 2
: Guilty Party ( 2017 ) 5 Development and production 6 Reception 6.1 Awards and accolades 7 International broadcast 8 References 9 External links Background (
edit ) Slasher employs a season - long anthology format , with an overarching mystery storyline that will be resolved upon the season 's conclusion .
</item 8>

<item 9>
Slasher (TV series)
Awards and accolades ( edit ) Year Award Nominee ( s ) Result 2017 5th Annual Canadian Screen Awards Best TV Movie or Limited Series Slasher Nominated Best
Direction in a Dramatic Program or Limited Series Nominated Best Writing in a Dramatic Program or Limited Series Nominated Best Performance by an Actor in a
Leading Role in a Dramatic Program or Limited Series Steve Byers Nominated Best Performance by an Actress in a Featured Supporting Role in a Dramatic Program or
Series Wendy Crewson Won International broadcast ( edit ) On May 25 , 2016 , the entire first season of Slasher became available to stream instantly on Netflix
US . On October 17 , 2017 , the full second season of Slasher became available to stream on Netflix in various countries . References ( edit ) Jump up ^
Brzoznowski , Kristin . `` Super Channel Sets April Premiere for Original Series Slasher '' . TVDrama.ws . World Screen ( WSN Inc . ) . Retrieved 6 March 2016 .
^ Jump up to : `` Chiller and Shaftesbury Announce Start of Production and Cast for `` Slasher '' `` .
</item 9>

<item 10>
Slasher (TV series)
Super Channel aired the show 's Canadian premiere on April 1 , 2016 . The series first season , which centered on a mysterious figure billed as `` The
Executioner '' terrorizing the fictional town of Waterbury , Canada , was filmed between July and October 2015 in Sudbury , Parry Sound , and Sault Ste . Marie
, Ontario . On May 1 , 2017 , Slasher began filming a second season in Orangeville , Ontario , Canada . The licensing rights for the second season was acquired
by Netflix . The season was released as a `` Netflix Original '' title in various countries on October 17 , 2017 . The second season titled Slasher : Guilty
Party is centered on a group of former summer camp counselors who return to an isolated campground in order to bury their sins of the past , before being
targeted , one by one , by an unknown killer .
</item 10>
=== END ITEM id=836b3525c2549248 ===

=== ITEM id=8403b8340e3e0db1 ===
Gold sentences and facts:
Sentence 0: Coordinates : 90 ° N 0 ° W ﻿ / ﻿ 90 ° N - 0 ° E ﻿ / 90 ; - 0
  Fact 0.0: The coordinates are listed as 90 degrees N, 0 degrees W.
  Fact 0.1: The coordinates are listed as 90 degrees N, 0 degrees E.
  Fact 0.2: The coordinates are listed as 90, -0.
Sentence 1: The North Pole is the northernmost point on the Earth , lying diametrically opposite the South Pole . It defines geodetic latitude 90 ° North , as well as the direction of true north . At the North Pole all directions point south ; all lines of longitude converge there , so its longitude can be defined as any degree value . Along tight latitude circles , counterclockwise is east and clockwise is west . The North Pole is at the center of the Northern Hemisphere .
  Fact 1.0: The North Pole is the northernmost point on the Earth.
  Fact 1.1: The North Pole lies diametrically opposite the South Pole.
  Fact 1.2: The North Pole defines geodetic latitude 90 degrees North.
  Fact 1.3: The North Pole defines the direction of true north.
  Fact 1.4: At the North Pole all directions point south.
  Fact 1.5: At the North Pole all lines of longitude converge.
  Fact 1.6: The longitude of the North Pole can be defined as any degree value.
  Fact 1.7: Along tight latitude circles, counterclockwise is east.
  Fact 1.8: Along tight latitude circles, clockwise is west.
  Fact 1.9: The North Pole is at the center of the Northern Hemisphere.

Retrieved items:
<item 1>
North Pole
Owing to its remoteness , the Pole is sometimes identified with a mysterious mountain of ancient Iranian tradition called Mount Qaf ( Jabal Qaf ) , the ``
farthest point of the earth '' . According to certain authors , the Jabal Qaf of Muslim cosmology is a version of Rupes Nigra , a mountain whose ascent , like
Dante 's climbing of the Mountain of Purgatory , represents the pilgrim 's progress through spiritual states . In Iranian theosophy , the heavenly Pole , the
focal point of the spiritual ascent , acts as a magnet to draw beings to its `` palaces ablaze with immaterial matter . '' See also ( edit ) Arctic portal
Arctic cooperation and politics Arctic Council Biome Celestial pole Ecliptic pole Inuit Circumpolar Council North Pole , Alaska Polaris Poles of astronomical
bodies Willem Barentsz Notes ( edit ) A.
</item 1>

<item 2>
North Pole
There is no permanent human presence at the North Pole and no particular time zone has been assigned . Polar expeditions may use any time zone that is
convenient , such as Greenwich Mean Time , or the time zone of the country from which they departed . Climate ( edit ) Arctic ice shrinkages of 2007 compared to
2005 and also compared to the 1979 -- 2000 average . Main article : Climate of the Arctic See also : Climate change in the Arctic The North Pole is
substantially warmer than the South Pole because it lies at sea level in the middle of an ocean ( which acts as a reservoir of heat ) , rather than at altitude
on a continental land mass . Despite being an ice cap , it shares some characteristics with a tundra climate ( ETf ) due to the July and August temperatures
peaking just above freezing . Winter temperatures at the North Pole can range from about − 50 to − 13 ° C ( − 58 to 9 ° F ) , averaging around − 31 ° C ( − 24 °
F ) .
</item 2>

<item 3>
North Pole
The direction of the Earth 's axial tilt , as well as its angle relative to the plane of the Earth 's orbit around the sun , remains very nearly constant over
the course of a year ( both change very slowly over long time periods ) . At northern midsummer the North Pole is facing towards the sun to its maximum extent .
As the year progresses and the Earth moves around the sun , the North Pole gradually turns away from the sun until at midwinter it is facing away from the Sun
to its maximum extent . A similar sequence is observed at the South Pole , with a six - month time difference . Time ( edit ) In most places on Earth , local
time is determined by longitude , such that the time of day is more - or-less synchronised to the position of the sun in the sky ( for example , at midday the
sun is roughly at its highest ) . This line of reasoning fails at the North Pole , where the sun rises and sets only once per year , and all lines of longitude
, and hence all time zones , converge . There is no permanent human presence at the North Pole and no particular time zone has been assigned .
</item 3>

<item 4>
North Pole
org/w/index.php?title=North_Pole&oldid=837303813 '' Categories : Extreme points of Earth Northern Canada Navigation Polar regions of the Earth Geography of the
Arctic Hidden categories : Webarchive template wayback links Articles with Russian - language external links CS1 maint : Uses authors parameter Wikipedia
indefinitely move - protected pages Use dmy dates from March 2015 Coordinates on Wikidata All articles lacking reliable references Articles lacking reliable
references from March 2016 All articles with unsourced statements Articles with unsourced statements from January 2017 Articles with unsourced statements from
December 2014 CS1 maint : Multiple names : authors list Wikipedia articles with VIAF identifiers Wikipedia articles with GND identifiers Talk Navigation
Contents About Wikipedia Wikivoyage Acèh Afrikaans Alemannisch አማርኛ Ænglisc Aragonés Asturianu Azərbaycanca تۆرکجه বাংলা Bân - lâm - gú Башҡортса Беларуская
Беларуская ( тарашкевіца ) ‎ Български Boarisch
</item 4>

<item 5>
North Pole
Finding the North Pole . The Lyons press . ISBN 1 - 59228 - 072 - 2 . CS1 maint : Multiple names : authors list ( link ) Pal , Christopher ( 2002 ) . The Oddest
Place on Earth : Rediscovering the North Pole . Writer 's Showcase . ISBN 0 - 595 - 21454 - 1 . Fairfield , William ( 1885 ) . Paradise Found a Cradle of the
Human Race at the North Pole a Study of the Prehistoric World . Houghton , Mifflin . ISBN 0 - 665 - 37759 - 2 . External links ( edit ) Wikimedia Commons has
media related to North Pole . Arctic Council The Northern Forum North Pole Web Cam FAQ on the Arctic and the North Pole Daylight , Darkness and Changing of the
Seasons at the North Pole Video of the Nuclear Icebreaker Yamal visiting the North Pole in 2001 Polar Discovery : North Pole Observatory Expedition Polar
exploration Arctic Ocean History Expeditions Research stations Farthest North North Pole Barentsz Hudson Marmaduke Carolus Parry North Magnetic Pole J. Ross
J.C.
</item 5>

<item 6>
North Pole
North Pole - wikipedia North Pole Jump to : navigation , search For other uses , see North Pole ( disambiguation ) . Not to be confused with North Magnetic Pole
. Coordinates : 90 ° N 0 ° W ﻿ / ﻿ 90 ° N - 0 ° E ﻿ / 90 ; - 0 An azimuthal projection showing the Arctic Ocean and the North Pole . The map also shows the 75th
parallel north and 60th parallel north . Sea ice at the North Pole in 2006 The North Pole , also known as the Geographic North Pole or Terrestrial North Pole ,
is ( subject to the caveats explained below ) defined as the point in the Northern Hemisphere where the Earth 's axis of rotation meets its surface . The North
Pole is the northernmost point on the Earth , lying diametrically opposite the South Pole . It defines geodetic latitude 90 ° North , as well as the direction
of true north . At the North Pole all directions point south ; all lines of longitude converge there , so its longitude can be defined as any degree value .
Along tight latitude circles , counterclockwise is east and clockwise is west .
</item 6>

<item 7>
North Pole
Along tight latitude circles , counterclockwise is east and clockwise is west . The North Pole is at the center of the Northern Hemisphere . While the South
Pole lies on a continental land mass , the North Pole is located in the middle of the Arctic Ocean amid waters that are almost permanently covered with
constantly shifting sea ice . This makes it impractical to construct a permanent station at the North Pole ( unlike the South Pole ) . However , the Soviet
Union , and later Russia , constructed a number of manned drifting stations on a generally annual basis since 1937 , some of which have passed over or very
close to the Pole . Since 2002 , the Russians have also annually established a base , Barneo , close to the Pole . This operates for a few weeks during early
spring . Studies in the 2000s predicted that the North Pole may become seasonally ice - free because of Arctic ice shrinkage , with timescales varying from 2016
to the late 21st century or later .
</item 7>

<item 8>
South Pole
The geographic coordinates of the South Pole are usually given simply as 90 ° S , since its longitude is geometrically undefined and irrelevant . When a
longitude is desired , it may be given as 0 ° . At the South Pole , all directions face north . For this reason , directions at the Pole are given relative to
`` grid north '' , which points northwards along the prime meridian . Along tight latitude circles , clockwise is east , and counterclockwise is west , opposite
to the North Pole . The Geographic South Pole is located on the continent of Antarctica ( although this has not been the case for all of Earth 's history
because of continental drift ) . It sits atop a featureless , barren , windswept and icy plateau at an altitude of 2,835 metres ( 9,301 ft ) above sea level ,
and is located about 1,300 km ( 800 mi ) from the nearest open sea at Bay of Whales . The ice is estimated to be about 2,700 metres ( 9,000 ft ) thick at the
Pole , so the land surface under the ice sheet is actually near sea level .
</item 8>

<item 9>
North Pole
Exploration ( edit ) See also : Arctic exploration , Farthest North , List of Arctic expeditions , and List of firsts in the Geographic North Pole Pre-1900 (
edit ) Gerardus Mercator 's map of the North Pole from 1595 C.G. Zorgdragers map of the North Pole from 1720 As early as the 16th century , many prominent
people correctly believed that the North Pole was in a sea , which in the 19th century was called the Polynya or Open Polar Sea . It was therefore hoped that
passage could be found through ice floes at favorable times of the year . Several expeditions set out to find the way , generally with whaling ships , already
commonly used in the cold northern latitudes . One of the earliest expeditions to set out with the explicit intention of reaching the North Pole was that of
British naval officer William Edward Parry , who in 1827 reached latitude 82 ° 45 ′ North . In 1871 the Polaris expedition , a US attempt on the Pole led by
Charles Francis Hall , ended in disaster .
</item 9>

<item 10>
North Pole
^ Data is from a Greenlandic Weather Station at 83 ° 38 ′ N 033 ° 22 ′ W ﻿ / ﻿ 83.633 ° N 33.367 ° W ﻿ / 83.633 ; - 33.367 ﻿ ( Greenlandic Weather Station )
located 709 km ( 441 mi ) from the North Pole . References ( edit ) Jump up ^ Black , Richard ( 8 April 2001 ) . New warning on Arctic sea ice melt . BBC Jump
up ^ Ljunggren , David ( 5 March 2009 ) . Arctic summer ice could vanish by 2013 : expert . Reuters Jump up ^ Russian sub plants flag at North Pole , Reuters ,
2 August 2007 Jump up ^ Андерсон , Уильям Роберт ( 1965 ) . `` '' Наутилус '' у Северного полюса `` . Воениздат . Retrieved 12 January 2012 . Jump up ^ Mouton ,
M.W. ( 1968 ) . The International Regime of the Polar Regions .
</item 10>
=== END ITEM id=8403b8340e3e0db1 ===

=== ITEM id=6af0793070c487f6 ===
Gold sentences and facts:
Sentence 0: The earliest predecessors of the eye were photoreceptor proteins that sense light , found even in unicellular organisms , called `` eyespots '' . Eyespots can only sense ambient brightness : they can distinguish light from dark , sufficient for photoperiodism and daily synchronization of circadian rhythms . They are insufficient for vision , as they can not distinguish shapes or determine the direction light is coming from . Eyespots are found in nearly all major animal groups , and are common among unicellular organisms , including euglena . The euglena 's eyespot , called a stigma , is located at its anterior end . It is a small splotch of red pigment which shades a collection of light sensitive crystals . Together with the leading flagellum , the eyespot allows the organism to move in response to light , often toward the light to assist in photosynthesis , and to predict day and night , the primary function of circadian rhythms . Visual pigments are located in the brains of more complex organisms , and are thought to have a role in synchronising spawning with lunar cycles . By detecting the subtle changes in night - time illumination , organisms could synchronise the release of sperm and eggs to maximise the probability of fertilisation .
  Fact 0.0: The earliest predecessors of the eye were photoreceptor proteins that sense light.
  Fact 0.1: The light-sensing photoreceptor proteins that were the earliest predecessors of the eye are found even in unicellular organisms.
  Fact 0.2: These early light-sensing structures are called 'eyespots'.
  Fact 0.3: Eyespots can only sense ambient brightness.
  Fact 0.4: Eyespots can distinguish light from dark.
  Fact 0.5: Sensing ambient brightness is sufficient for photoperiodism.
  Fact 0.6: Sensing ambient brightness is sufficient for daily synchronization of circadian rhythms.
  Fact 0.7: Eyespots are insufficient for vision.
  Fact 0.8: Eyespots cannot distinguish shapes.
  Fact 0.9: Eyespots cannot determine the direction light is coming from.
  Fact 0.10: Eyespots are found in nearly all major animal groups.
  Fact 0.11: Eyespots are common among unicellular organisms.
  Fact 0.12: Euglena is a unicellular organism with an eyespot.
  Fact 0.13: The euglena's eyespot is called a stigma.
  Fact 0.14: The euglena's stigma is located at its anterior end.
  Fact 0.15: The euglena's stigma is a small splotch of red pigment.
  Fact 0.16: The euglena's stigma shades a collection of light-sensitive crystals.
  Fact 0.17: Together with the leading flagellum, the euglena's eyespot allows the organism to move in response to light.
  Fact 0.18: The euglena often moves toward the light to assist in photosynthesis.
  Fact 0.19: The euglena's eyespot allows the organism to predict day and night.
  Fact 0.20: Predicting day and night is the primary function of circadian rhythms.
  Fact 0.21: Visual pigments are located in the brains of more complex organisms.
  Fact 0.22: Visual pigments are thought to have a role in synchronising spawning with lunar cycles.
  Fact 0.23: Organisms could detect subtle changes in night-time illumination.
  Fact 0.24: By detecting night-time illumination changes, organisms could synchronise the release of sperm and eggs.
  Fact 0.25: Synchronising the release of sperm and eggs maximises the probability of fertilisation.

Retrieved items:
<item 1>
Evolution of the eye
Monogamy Morality Mosaic evolution Multicellularity Sexual reproduction Gamete differentiation / sexes Life cycles / nuclear phases Mating types Sex -
determination Snake venom Tempo and modes Gradualism / Punctuated equilibrium / Saltationism Micromutation / Macromutation Uniformitarianism / Catastrophism
Speciation Allopatric Anagenesis Catagenesis Cladogenesis Cospeciation Ecological Hybrid Parapatric Peripatric Reinforcement Sympatric History Renaissance and
Enlightenment Transmutation of species Charles Darwin On the Origin of Species History of paleontology Transitional fossil Blending inheritance Mendelian
inheritance The eclipse of Darwinism Modern synthesis History of molecular evolution Extended evolutionary synthesis Philosophy Darwinism Alternatives
Catastrophism Lamarckism Orthogenesis Mutationism Saltationism Structuralism Spandrel Theistic Vitalism Teleology in biology Related Biogeography Ecological
genetics Molecular evolution Phylogenetics Tree Polymorphism Protocell Systematics Commons Portal WikiProject Vision in animals Vision Birds Cats Chameleons
Dinosaurs Dogs Fishes Horses Toads Eyes Arthropod eye Compound eye Eagle eye Eye shine Simple eye in invertebrates Mammalian eye Human eye Mollusc eye
Cephalopod eye Gastropod eye
</item 1>

<item 2>
Evolution of the eye
`` How Complexity Originates : The Evolution of Animal Eyes '' . Annual Review of Ecology , Evolution , and Systematics. 46 : 237 -- 260 . doi : 10.1146 /
annurev - ecolsys - 110512 - 135907 . Ed Young ; photographs by David Liittschwager ( February 2016 ) . `` Inside the Eye : Nature 's Most Exquisite Creation ''
. National Geographic . 229 ( 2 ) : 30 -- 57 . External links ( edit ) `` Evolution of the Eye '' . WGBH Educational Foundation and Clear Blue Sky Productions .
PBS . 2001 . Creationism Disproved ? Video from the National Center for Science Education on the evolution of the eye Evolution : Education and Outreach Special
Issue : Evolution and Eyes volume 1 , number 4 , October 2008 , pages 351 - 559 . ISSN 1936 - 6426 ( Print ) 1936 - 6434 ( Online ) Evolutionary biology
Evolutionary history of life Index of evolutionary biology articles Introduction Outline of evolution Timeline of evolution Evolution Abiogenesis Adaptation
Adaptive radiation Cladistics
</item 2>

<item 3>
Evolution of the eye
biology articles Introduction Outline of evolution Timeline of evolution Evolution Abiogenesis Adaptation Adaptive radiation Cladistics Coevolution Common
descent Convergence Divergence Earliest known life forms Evidence of common descent Extinction Event Gene - centered view Homology Last universal common
ancestor Macroevolution Microevolution Origin of life Panspermia Parallel evolution Prehistoric Autopsy Speciation Taxonomy Population genetics Biodiversity
Gene flow Genetic drift Mutation Natural selection Variation Development Canalisation Evolutionary developmental biology Inversion Modularity Phenotypic
plasticity Of taxa Birds origin Brachiopods Molluscs Cephalopods Dinosaurs Fish Fungi Insects butterflies Life Mammals cats canids wolves dogs hyenas dolphins
and whales horses primates humans lemurs sea cows Plants Reptiles Spiders Tetrapods Viruses influenza Of organs Cell DNA Flagella Eukaryotes symbiogenesis
chromosome endomembrane system mitochondria nucleus plastids In animals eye hair auditory ossicle nervous system brain Of processes Aging Death Programmed cell
death Avian flight Biological complexity Cooperation Color vision in primates Emotion Empathy Ethics Eusociality Immune system Metabolism Monogamy Morality
Mosaic evolution Multicellularity Sexual reproduction Gamete differentiation / sexes
</item 3>

<item 4>
Evolution of the eye
Evolution of the eye - wikipedia Evolution of the eye Jump to : navigation , search Major stages in the evolution of the eye in vertebrates . The evolution of
the eye is attractive to study , because the eye distinctively exemplifies an analogous organ found in many animal forms . Complex , image - forming eyes have
evolved independently several times . The evolutionary geneticist Walter Gering holds a different opinion . According his theory , the genes engaged with
photoreceptor cells ( so called Pax 6 genes ) appeared probably at once in the life history , at a very early stage of the evolution . Probably , the complex ,
image forming kinds of eyes are at least five : Chidaria - lens eye , complex eye ; Molluscs - lens eye , everse retina ; Annelids - lens eye , everse retina ;
Arthropods - facet eye ; Vertebrate - lens eye , inverse retina Complex eyes appeared first within the few million years of the Cambrian explosion . From before
the Cambrian , no evidence of eyes has survived , but diverse eyes are known from the Burgess shale of the Middle Cambrian , and from the slightly older Emu Bay
Shale .
</item 4>

<item 5>
Evolution of the eye
In both vertebrates and octopodes , 1 represents the retina , 2 represents the nerve fibers , and 3 represents the optic nerve . Main article : Evolutionary
baggage The eyes of many animals record their evolutionary history in their contemporary anatomy . The vertebrate eye , for instance , is built `` backwards and
upside down '' , requiring `` photons of light to travel through the cornea , lens , aqueous fluid , blood vessels , ganglion cells , amacrine cells ,
horizontal cells , and bipolar cells before they reach the light - sensitive rods and cones that transduce the light signal into neural impulses , which are
then sent to the visual cortex at the back of the brain for processing into meaningful patterns . '' While such a construct has some drawbacks , it also allows
the outer retina of the vertebrates to sustain higher metabolic activities as compared to the non-inverted design . It also allowed for the evolution of the
choroid layer , including the retinal pigment epithelial ( RPE ) cells , which play an important role in protecting the photoreceptive cells from photo -
oxidative damage .
</item 5>

<item 6>
Evolution of the eye
149 Jump up ^ Lee , M.S.Y. ; Jago , J.B. ; Garcia - Bellido , D.C. ; Edgecombe , G.E. ; Gehling , J. G ; Paterson , J.R. ( 2011 ) . `` Modern optics in
exceptionally preserved eyes of Early Cambrian arthropods from Australia '' . Nature . 474 : 631 -- 634 . doi : 10.1038 / nature10097 . PMID 21720369 . ^ Jump
up to : Darwin , Charles ( 1859 ) . On the Origin of Species . London : John Murray . Jump up ^ Gehring WJ ( 2005 ) . `` New perspectives on eye development and
the evolution of eyes and photoreceptors '' . J. Hered. 96 ( 3 ) : 171 -- 84 . doi : 10.1093 / jhered / esi027 . PMID 15653558 . Jump up ^ Nilsson , D. - E. (
2013 ) . `` Eye evolution and its functional basis '' . Visual Neuroscience.
</item 6>

<item 7>
Evolution of the eye
`` Eye evolution and its functional basis '' . Visual Neuroscience. 30 : 5 -- 20 . doi : 10.1017 / s0952523813000035 . ^ Jump up to : Nilsson , D. - E. ; Pelger
, S. ( 1994 ) . `` A pessimistic estimate of the time required for an eye to evolve '' . Proceedings of the Royal Society B : Biological Sciences . 256 : 53 --
58 . doi : 10.1098 / rspb. 1994.0048 . PMID 8008757 . Jump up ^ Young , G.C. ( 2008 ) . `` Early evolution of the vertebrate eye -- fossil evidence '' . Evo Edu
Outreach. 1 : 427 -- 438 . doi : 10.1007 / s12052 - 008 - 0087 - y . Jump up ^ Parker , A.R. ( 2009 ) . `` On the origin of optics '' . Optics & Laser
Technology . 43 ( 2 ) : 323 -- 329 .
</item 7>

<item 8>
Evolution of the eye
This suggests that when the two lineages diverged in the Precambrian , they had only very primitive light receptors , which developed into more complex eyes
independently . Early eyes ( edit ) The basic light - processing unit of eyes is the photoreceptor cell , a specialized cell containing two types of molecules
in a membrane : the opsin , a light - sensitive protein , surrounding the chromophore , a pigment that distinguishes colors . Groups of such cells are termed ``
eyespots '' , and have evolved independently somewhere between 40 and 65 times . These eyespots permit animals to gain only a very basic sense of the direction
and intensity of light , but not enough to discriminate an object from its surroundings . Developing an optical system that can discriminate the direction of
light to within a few degrees is apparently much more difficult , and only six of the thirty - some phyla possess such a system . However , these phyla account
for 96 % of living species . The planarian has `` cup '' eyespots that can slightly distinguish light direction .
</item 8>

<item 9>
Evolution of the eye
PMID 25734540 . Greuet , C ( 1968 ) . `` Organisation ultrastructurale de l'ocelle de deux Peridiniens Warnowiidae , Erythropsis pavillardi Kofoid et Swezy et
Warnowia pulchra Schiller '' . Protistologica. 4 : 209 -- 230 . Gregory S. Gavelis , Shiho Hayakawa , Richard A. White III , Takashi Gojobori , Curtis A. Suttle
, Patrick J. Keeling , Brian S. Leander ( 2015 ) . `` Eye - like ocelloids are built from different endosymbiotically acquired components '' . Nature . 523 :
204 -- 7 . Bibcode : 2015Natur. 523 ... 204G . doi : 10.1038 / nature14593 . PMID 26131935 . CS1 maint : Uses authors parameter ( link ) Oakley , Todd H. ;
Speiser , Daniel I. ( 2015 ) . `` How Complexity Originates : The Evolution of Animal Eyes '' .
</item 9>

<item 10>
Evolution of the eye
Nilsson and S. Pelger estimated in a classical paper how many generations are needed to evolve a complex eye in vertebrates . Another researcher , G.C. Young ,
has used the fossil record to infer evolutionary conclusions , based on the structure of eye orbits and openings in fossilized skulls for blood vessels and
nerves to go through . All this adds to the growing amount of evidence that supports Darwin 's theory . Rate of evolution ( edit ) The first fossils of eyes
found to date are from the lower Cambrian period ( about 540 million years ago ) . The lower Cambrian had a burst of apparently rapid evolution , called the ``
Cambrian explosion '' . One of the many hypotheses for `` causes '' of the Cambrian explosion is the `` Light Switch '' theory of Andrew Parker : It holds that
the evolution of eyes started an arms race that accelerated evolution . Before the Cambrian explosion , animals may have sensed light , but did not use it for
fast locomotion or navigation by vision . The rate of eye evolution is difficult to estimate , because the fossil record , particularly of the lower Cambrian ,
is poor .
</item 10>
=== END ITEM id=6af0793070c487f6 ===
