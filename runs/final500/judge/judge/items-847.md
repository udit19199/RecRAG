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

=== ITEM id=9401f14f90c03236 ===
Gold sentences and facts:
Sentence 0: Character Actor Job House Song ( s ) Miss Hoolie Julie Wilson Nimmo Nursery teacher Green house Nursery Song : Everybody everyone Day off song : What do you want to do today ( Series 1 -- 2 ) Strike up the band ( Series 3 -- 4 ) Archie the Inventor Miles Jupp Inventor Pink castle I 'm Archie the Inventor ( Series 1 - 4 ) Great Inventions , Groovy Solutions ( Series 3 -- 4 ) Josie Jump Buki Akib ( series 1 - 2 ) Kasia Haddad ( Series 3 -- 4 ) Fitness instructor Yellow house Josie Jump is My Name ( Jump a Little Higher ) ( Series 1 - 4 ) Cheer You Up ( Series 3 -- 4 ) Edie McCredie Juliet Cadzow Bus driver Blue garage / house When I Honk My Horn ( Series 1 - 4 ) Let Me Take You on a Journey ( Series 3 -- 4 ) PC Plum Andrew Agnew Policeman Black and white police station I 'm PC Plum ( Series 1 -- 4 ) Follow the Clue ( Series 3 -- 4 ) Spencer the Painter Rodd Christensen Painter / Musician Orange house Climbing Up My Musical Ladder ( Series 1 -- 4 ) If You Need a Little Rhythm ( Series 3 -- 4 ) Suzie Sweet Mary Riggans Shop and café workers Red shop I 'm Suzie Sweet , I 'm Penny Pocket ( Series 1 -- 4 ) Suzie 's Cooking ( Series 3 -- 4 ) Penny Pocket Kim Tserkezie I 'm Suzie Sweet , I 'm Penny Pocket ( Series 1 -- 4 ) Sort It ( Series 3 -- 4 )
  Fact 0.0: Miss Hoolie is played by Julie Wilson Nimmo.
  Fact 0.1: Miss Hoolie's job is nursery teacher.
  Fact 0.2: Miss Hoolie's house is the Green house (Nursery).
  Fact 0.3: Miss Hoolie's Nursery Song is Everybody everyone.
  Fact 0.4: Miss Hoolie's Day off song is What do you want to do today (Series 1-2).
  Fact 0.5: Miss Hoolie's Day off song is Strike up the band (Series 3-4).
  Fact 0.6: Archie the Inventor is played by Miles Jupp.
  Fact 0.7: Archie the Inventor's job is inventor.
  Fact 0.8: Archie the Inventor's house is the Pink castle.
  Fact 0.9: Archie the Inventor's song is I'm Archie the Inventor (Series 1-4).
  Fact 0.10: Archie the Inventor's song is Great Inventions, Groovy Solutions (Series 3-4).
  Fact 0.11: Josie Jump is played by Buki Akib in Series 1-2.
  Fact 0.12: Josie Jump is played by Kasia Haddad in Series 3-4.
  Fact 0.13: Josie Jump's job is fitness instructor.
  Fact 0.14: Josie Jump's house is the Yellow house.
  Fact 0.15: Josie Jump's song is Josie Jump is My Name (Jump a Little Higher) (Series 1-4).
  Fact 0.16: Josie Jump's song is Cheer You Up (Series 3-4).
  Fact 0.17: Edie McCredie is played by Juliet Cadzow.
  Fact 0.18: Edie McCredie's job is bus driver.
  Fact 0.19: Edie McCredie's house is the Blue garage / house.
  Fact 0.20: Edie McCredie's song is When I Honk My Horn (Series 1-4).
  Fact 0.21: Edie McCredie's song is Let Me Take You on a Journey (Series 3-4).
  Fact 0.22: PC Plum is played by Andrew Agnew.
  Fact 0.23: PC Plum's job is policeman.
  Fact 0.24: PC Plum's house is the Black and white police station.
  Fact 0.25: PC Plum's song is I'm PC Plum (Series 1-4).
  Fact 0.26: PC Plum's song is Follow the Clue (Series 3-4).
  Fact 0.27: Spencer the Painter is played by Rodd Christensen.
  Fact 0.28: Spencer the Painter's job is painter and musician.
  Fact 0.29: Spencer the Painter's house is the Orange house.
  Fact 0.30: Spencer the Painter's song is Climbing Up My Musical Ladder (Series 1-4).
  Fact 0.31: Spencer the Painter's song is If You Need a Little Rhythm (Series 3-4).
  Fact 0.32: Suzie Sweet is played by Mary Riggans.
  Fact 0.33: Suzie Sweet's job is shop and café worker.
  Fact 0.34: Suzie Sweet's house is the Red shop.
  Fact 0.35: Suzie Sweet's song is I'm Suzie Sweet, I'm Penny Pocket (Series 1-4).
  Fact 0.36: Suzie Sweet's song is Suzie's Cooking (Series 3-4).
  Fact 0.37: Penny Pocket is played by Kim Tserkezie.
  Fact 0.38: Penny Pocket's song is I'm Suzie Sweet, I'm Penny Pocket (Series 1-4).
  Fact 0.39: Penny Pocket's song is Sort It (Series 3-4).

Retrieved items:
<item 1>
Balamory
I 'm Penny Pocket ( Series 1 -- 4 ) Suzie 's Cooking ( Series 3 -- 4 ) Penny Pocket Kim Tserkezie I 'm Suzie Sweet , I 'm Penny Pocket ( Series 1 -- 4 ) Sort It
( Series 3 -- 4 ) The nursery teacher Miss Hoolie is the framing character and narrator for each episode . She introduces the programme and the problem which
needs to be solved , provides a recap in the middle , and summarises the story at the end of the episode . Archie the Inventor builds zany creations from
household objects , such as boxes , bottles , yoghurt pots , string , and similar items that young children might typically use in craft projects . His
inventions are intended to be useful , but sometimes backfire . Josie Jump is the fitness instructor at the primary school next door to Miss Hoolie 's nursery
school . She plays and coaches all types of gymnastics , sports , and dance , and encourages the young viewers to exercise . She is also talented at
storytelling through expressive movement and dance . Edie McCredie drives the nursery school bus .
</item 1>

<item 2>
Balamory
28 -- 30 minutes . Specials Production company ( s ) BBC Scotland Distributor BBC Release Original network CBeebies Picture format 16 : 9 Widescreen Audio
format Stereo Original release 10 February 2002 -- 14 April 2005 External links Website www.bbc.co.uk/cbeebies/shows/balamory Balamory is a Scottish live action
television series on British television ( BBC One , BBC Two and CBeebies ) for pre school children , about a fictional small island community called Balamory in
Scotland . It was produced between 10 February 2002 and 14 April 2005 by BBC Scotland , with 254 episodes made ( including a DVD exclusive Christmas episode ) .
The series was created by Brian Jameson . Balamory was filmed mostly in Tobermory on the Isle of Mull , with the exception of scenes at Archie 's castle filmed
at Fenton Tower in North Berwick , and other scenes , such as Camden and the nursery filmed in Glasgow . A select series of episodes was shown for two brief
seasons in the United States on the Discovery Kids channel . It was also shown on TLC .
</item 2>

<item 3>
Balamory
Groovy Solutions ( Series 3 -- 4 ) Josie Jump Buki Akib ( series 1 - 2 ) Kasia Haddad ( Series 3 -- 4 ) Fitness instructor Yellow house Josie Jump is My Name (
Jump a Little Higher ) ( Series 1 - 4 ) Cheer You Up ( Series 3 -- 4 ) Edie McCredie Juliet Cadzow Bus driver Blue garage / house When I Honk My Horn ( Series 1
- 4 ) Let Me Take You on a Journey ( Series 3 -- 4 ) PC Plum Andrew Agnew Policeman Black and white police station I 'm PC Plum ( Series 1 -- 4 ) Follow the
Clue ( Series 3 -- 4 ) Spencer the Painter Rodd Christensen Painter / Musician Orange house Climbing Up My Musical Ladder ( Series 1 -- 4 ) If You Need a Little
Rhythm ( Series 3 -- 4 ) Suzie Sweet Mary Riggans Shop and café workers Red shop I 'm Suzie Sweet ,
</item 3>

<item 4>
Balamory
It was also shown on TLC . In Ireland , episodes were shown until recently on weekdays on RTÉ Two 's The Den and continue to be shown on The Den 's replacement
, RTÉjr . Tobermory , Mull , the setting for the fictional town of Balamory Edie McCredie 's bus ( LDV Convoy ) in the Glasgow Museum of Transport Contents (
hide ) 1 Characters 2 Episodes 2.1 Episode guide 3 Audience 4 Songs 5 Production 6 Retirement 7 Awards 8 References 9 External links Characters ( edit ) There
are eight central characters in Balamory . Each character wears a distinctive colour of clothing like pink or yellow , and lives in a house of the same colour
like Archie lives in a pink castle , with the exception of Penny Pocket ( who shares the red house / shop with Suzie Sweet , and wears blue ) and PC Plum ( who
lives in the white house but wears a police uniform ) . Each character also has one or more individual songs .
</item 4>

<item 5>
Balamory
Each character also has one or more individual songs . Character Actor Job House Song ( s ) Miss Hoolie Julie Wilson Nimmo Nursery teacher Green house Nursery
Song : Everybody everyone Day off song : What do you want to do today ( Series 1 -- 2 ) Strike up the band ( Series 3 -- 4 ) Archie the Inventor Miles Jupp
Inventor Pink castle I 'm Archie the Inventor ( Series 1 - 4 ) Great Inventions ,
</item 5>

<item 6>
Balamory
Balamory - wikipedia Balamory Jump to : navigation , search This article needs additional citations for verification . Please help improve this article by
adding citations to reliable sources . Unsourced material may be challenged and removed . ( December 2010 ) ( Learn how and when to remove this template message
) Balamory Created by Brian Jameson Starring Julie Wilson Nimmo Miles Jupp Kasia Haddad Buki Akib Juliet Cadzow Andrew Agnew Rodd Christensen Mary Riggans Kim
Tserkezie Country of origin United Kingdom Original language ( s ) English No. of episodes 254 , including Christmas special . Production Producer ( s ) Brian
Jameson Running time 19 -- 21 minutes . 28 -- 30 minutes .
</item 6>

<item 7>
Balamory
Edie McCredie drives the nursery school bus . She has travelled all over the world , and has a variety of photographs and souvenirs to talk about . She is also
an expert driver and motor mechanic , and occasionally gets exasperated with other road users . Police officer PC Plum aspires to be a master detective , but
Balamory is a crime free village . He spends most of his time watching wildlife and using his detective skills , to figure out what the animals have been doing
. He is also particularly interested in rules , regulations , and safety . Spencer is officially responsible for painting the coloured houses of Balamory , but
also paints pictures and creates music . His painter 's ladder is musical , each rung sounding a different note . Spencer is also the only American to reside in
Balamory and he is black . Suzie Sweet and Penny Pocket together run the village shop and cafe . Suzie Sweet is an older , grandmotherly figure who loves to
tell customers about the items in the shop . Penny Pocket is a young woman who uses a wheelchair ; she has a talent for mathematics .
</item 7>

<item 8>
Balamory
She is advised to visit PC Plum , who advises her to feed it vegetables . She then goes to the shop run by Pocket and Sweet to buy rabbit food , and finally
brings the rabbit to Miss Hoolie 's nursery for the children to feed and pat it . Each episode of Balamory contains at least five songs : `` What 's the Story
in Balamory ? '' , the theme song Miss Hoolie 's opening song , either `` Everybody , Everyone '' if the nursery school is open , or one of the two `` day off
'' songs otherwise `` Which Coloured House Are We Going To ? '' , sung by the main character after presenting his or her problem Character - specific songs ,
sung by the characters visited by the main character Episode guide ( edit ) Series One Series Two Series Three Series Four Audience ( edit ) The programme was
aimed at pre schoolers . The programme 's creator had intended it as a `` soap opera for children '' . The show can also be seen in a humorous light . The
characters in Balamory are somewhat naive and light hearted , often appearing to be children trapped in adult bodies with adult roles .
</item 8>

<item 9>
Balamory
The storyline revolves around simple problems designed to challenge pre schoolers ' minds . Songs ( edit ) There are many songs in Balamory , mainly sung by the
actors . These songs are used consistently throughout the series , when corresponding characters or circumstances are introduced . In the run starting from
September 2004 , new songs were introduced . These songs are longer and take up more time in the episode , so the old versions were still often used . In
addition to the character specific songs mentioned in the `` Characters '' section , there are two general songs : `` What 's the Story in Balamory ? '' is the
show 's theme song , sung over the opening titles and played as an instrumental over the closing credits . `` Which Coloured House Are We Going To ? '' is sung
by the main character of each episode after introducing that episode 's problem . Production ( edit ) Very few episodes feature actors other than the main
characters and the nursery children . Terry Wogan made a guest appearance in one episode ( The Game Show ) as a television director .
</item 9>

<item 10>
Balamory
Penny Pocket is a young woman who uses a wheelchair ; she has a talent for mathematics . Episodes ( edit ) A typical episode follows this general plan : The
opening credits show the town of `` Balamory '' with its brightly coloured houses . Miss Hoolie opens the nursery school , greets the nursery children , and
talks about that day 's weather . One of the other characters ( who will be the main character of that episode ) tells Miss Hoolie about a problem . This
character then sings `` Which Coloured House Are We Going To ? '' , asking the young viewers to guess which of the Balamory characters will be most helpful in
solving the problem . The main character then proceeds to visit several other characters until the problem is solved . Miss Hoolie and the main character
provide a summary of the story at each point , and a final recap at the end of the episode . For example , in one episode Josie Jump has been asked to look
after a neighbour 's rabbit , but does not know what it needs . She is advised to visit PC Plum , who advises her to feed it vegetables .
</item 10>
=== END ITEM id=9401f14f90c03236 ===

=== ITEM id=8e234d4480a28b15 ===
Gold sentences and facts:
Sentence 0: The Proud Family is a soundtrack album for the show of the same name . A combination of both original tunes by the characters in the show , popular hits by popular music artists such as Alicia Keys , Jhené Aiko , India Arie , and Solange Knowles & Destiny 's Child ( who perform the theme song ) , and classic soul music from artists such as Aretha Franklin and The O'Jays .
  Fact 0.0: The Proud Family is a soundtrack album.
  Fact 0.1: The Proud Family soundtrack album is for the show of the same name.
  Fact 0.2: The album combines original tunes by the characters in the show.
  Fact 0.3: The album combines popular hits by popular music artists such as Alicia Keys, Jhené Aiko, India Arie, and Solange Knowles & Destiny's Child.
  Fact 0.4: Solange Knowles & Destiny's Child perform the theme song.
  Fact 0.5: The album combines classic soul music from artists such as Aretha Franklin and The O'Jays.

Retrieved items:
<item 1>
The excerpts cover several unrelated topics: Charlotte radio hosts John Boy and Billy’s syndicated comedy show; pizza chains and John’s Incredible Pizza
Company, a 1997-founded buffet and entertainment business with themed dining rooms; the 1978 film *Harper Valley PTA*; and *The Partridge Family*, ABC’s 1970–74
musical sitcom, its cast, hit recordings, spin-off, awards, and releases. Other items discuss *The Proud Family* soundtrack, actor-singer
</item 1>

<item 2>
The Proud Family (soundtrack)
The Proud Family ( soundtrack ) - Wikipedia The Proud Family ( soundtrack ) Jump to : navigation , search This article does not cite any sources . Please help
improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( December 2009 ) ( Learn how and when to
remove this template message ) The Proud Family Soundtrack album by Various Artists Released April 20 , 2004 August 16 , 2005 ( Re-release ) Genre Soul , hip
hop , hip hop soul Label Walt Disney The Proud Family is a soundtrack album for the show of the same name . A combination of both original tunes by the
characters in the show , popular hits by popular music artists such as Alicia Keys , Jhené Aiko , India Arie , and Solange Knowles & Destiny 's Child ( who
perform the theme song ) , and classic soul music from artists such as Aretha Franklin and The O'Jays . References ( edit ) This soundtrack - related article is
a stub . You can help Wikipedia by expanding it .
</item 2>

<item 3>
The excerpts cover Charlotte radio duo John Boy and Billy; John’s Incredible Pizza Company; *Harper Valley PTA*; and *The Partridge Family*, including its cast,
music, spin-off, and awards. They summarize *The Proud Family* soundtrack and Seinfeld’s “The Chinese Restaurant” (1991), in which Jerry, George, and Elaine
wait for a table. Most detail concerns *Sing* (2016), Buster Moon’s singing contest, its contestants, soundtrack, and
</item 3>

<item 4>
The Partridge Family
The Partridge Family - wikipedia The Partridge Family Jump to : navigation , search The Partridge Family Genre Musical sitcom Created by Bernard Slade Starring
Shirley Jones David Cassidy Danny Bonaduce Susan Dey Suzanne Crough Jeremy Gelbwaks Dave Madden Brian Forster Theme music composer Diane Hilderbrand Danny
Janssen Wes Farrell Opening theme `` When We 're Singin ' '' ( 1970 -- 71 ) `` C'mon , Get Happy '' ( 1971 -- 74 ) Composer ( s ) George Duning Benny Golson
Warren Barker Hugo Montenegro Shorty Rogers Country of origin United States Original language ( s ) English No. of seasons No. of episodes 96 ( list of episodes
) Production Executive producer ( s ) Bob Claver Producer ( s ) William Bickley Paul Junger Witt Dale McRaven Larry Rosen Mel Swope Cinematography Fred Jackman
, Jr .
</item 4>

<item 5>
*The Proud Family* soundtrack is a Walt Disney release featuring music from the animated series. The album blends original songs by the show’s characters with
popular tracks by artists including Alicia Keys, Jhené Aiko, India Arie, and Solange Knowles and Destiny’s Child, who perform the theme song, alongside classic
soul by Aretha Franklin and The O’Jays. It spans soul, hip-hop, and hip-hop soul, and was released April 20, 2004,
</item 5>

<item 6>
You'll Never Find Another Love Like Mine
in which Lou Rawls himself sings it to Damon Wayans during a colonoscopy , That ' 70s Show ( 1998 -- 2006 ) , South Park ( 1997 -- ) , Two and a Half Men ( 2003
-- 2015 ) , and Psych ( 2006 -- 2014 ) . The song could also be heard in an episode of The Proud Family , where Lou Rawls sings and dances with Penny Proud . In
addition , it was performed by the singer in doll form in an Action League Now ! segment on KaBlam ! . The song `` If You Could Love Me '' by Edwyn Collins uses
many of the same chords , and has a similar sound . Rocco DiSpirito and Karina Smirnoff also danced the rumba to this song in week 2 of season 7 of Dancing with
the Stars .
</item 6>

<item 7>
The Proud Family (soundtrack)
You can help Wikipedia by expanding it . Retrieved from `` https://en.wikipedia.org/w/index.php?title=The_Proud_Family_(soundtrack)&oldid=638462881 ''
Categories : Television animation soundtracks 2004 soundtracks Hip hop soundtracks Soul soundtracks Walt Disney Records soundtracks Soundtrack stubs Hidden
categories : Articles lacking sources from December 2009 All articles lacking sources Articles with hAudio microformats Music infoboxes with deprecated
parameters All stub articles Talk About Wikipedia Add links This page was last edited on 17 December 2014 , at 07 : 09 . Text is available under the Creative
Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a
registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 7>

<item 8>
*The Partridge Family* used two theme songs. Season 1 featured “When We’re Singin’,” whose lyrics describe the family helping their mother by making music, with
Danny getting Reuben to sell their song. In later seasons, it was replaced by “C’mon Get Happy.” Under Farrell’s direction, the new song used the original tune
with revised lyrics, including a verse about traveling together and spreading love.
</item 8>

<item 9>
The Partridge Family
More than one song was used as the theme song . Season 1 episodes feature the song `` When We 're Singin ' '' , which was replaced in subsequent seasons with ``
C'mon Get Happy '' . `` Five of us , and Mom working all day , we knew we could help her if our music would pay , Danny got Reuben to sell our song , and it
really came together when Mom sang along ... '' ( from `` When We 're Singin ' '' ) Later , under Farrell 's direction , the new version had new lyrics sung to
the `` When We 're Singin ' '' tune . With the new chorus finalized , `` C'mon Get Happy '' showcased the new verse : `` We had a dream , we 'd go travelin '
together , We 'd spread a little lovin ' then we 'd keep movin ' on .
</item 9>

<item 10>
*The Partridge Family* was an American musical sitcom created by Bernard Slade, with 96 episodes across an unspecified number of seasons. It starred Shirley
Jones, David Cassidy, Danny Bonaduce, Susan Dey, Suzanne Crough, Jeremy Gelbwaks, Dave Madden, and Brian Forster. Its opening themes were “When We’re Singin’”
(1970–71) and “C’mon, Get Happy” (1971–74). Bob Claver was executive
</item 10>
=== END ITEM id=8e234d4480a28b15 ===

=== ITEM id=82081dc9df9ffb17 ===
Gold sentences and facts:
Sentence 0: John Brown 's raid on Harper 's Ferry ( also known as John Brown 's raid or The raid on Harper 's Ferry ) was an effort by armed abolitionist John Brown to initiate an armed slave revolt in 1859 by taking over a United States arsenal at Harpers Ferry , Virginia . Brown 's party of 22 was defeated by a company of U.S. Marines , led by First Lieutenant Israel Greene . Colonel Robert E. Lee was in overall command of the operation to retake the arsenal . John Brown had originally asked Harriet Tubman and Frederick Douglass , both of whom he had met in his transformative years as an abolitionist in Springfield , Massachusetts , to join him in his raid , but Tubman was prevented by illness and Douglass declined , as he believed Brown 's plan would fail .
  Fact 0.0: John Brown's raid on Harper's Ferry is also known as John Brown's raid.
  Fact 0.1: John Brown's raid on Harper's Ferry is also known as The raid on Harper's Ferry.
  Fact 0.2: John Brown's raid on Harper's Ferry was an effort by armed abolitionist John Brown to initiate an armed slave revolt in 1859.
  Fact 0.3: The raid aimed to start the armed slave revolt by taking over a United States arsenal at Harpers Ferry, Virginia.
  Fact 0.4: Brown's party of 22 was defeated by a company of U.S. Marines.
  Fact 0.5: The company of U.S. Marines that defeated Brown's party was led by First Lieutenant Israel Greene.
  Fact 0.6: Colonel Robert E. Lee was in overall command of the operation to retake the arsenal.
  Fact 0.7: John Brown had originally asked Harriet Tubman to join him in the raid.
  Fact 0.8: John Brown had originally asked Frederick Douglass to join him in the raid.
  Fact 0.9: John Brown had met Harriet Tubman and Frederick Douglass during his transformative years as an abolitionist in Springfield, Massachusetts.
  Fact 0.10: Harriet Tubman was prevented from joining the raid by illness.
  Fact 0.11: Frederick Douglass declined to join the raid.
  Fact 0.12: Frederick Douglass believed Brown's plan would fail.
Sentence 1: By 3 : 30 that afternoon , President James Buchanan ordered a company of U.S. Marines ( the only government troops in the immediate area ) to march on Harpers Ferry under the command of Brevet Colonel Robert E. Lee , lieutenant colonel of the 2nd U.S. Cavalry Regiment. Lee had been on leave from his regiment , stationed in Texas , when he was hastily called to lead the detachment and had to command it while wearing his civilian clothes .
  Fact 1.0: By 3:30 that afternoon, President James Buchanan ordered a company of U.S. Marines to march on Harpers Ferry.
  Fact 1.1: The Marines were the only government troops in the immediate area.
  Fact 1.2: The Marines marched on Harpers Ferry under the command of Brevet Colonel Robert E. Lee.
  Fact 1.3: Robert E. Lee was lieutenant colonel of the 2nd U.S. Cavalry Regiment.
  Fact 1.4: Lee had been on leave from his regiment when he was hastily called to lead the detachment.
  Fact 1.5: Lee's regiment was stationed in Texas.
  Fact 1.6: Lee had to command the detachment while wearing his civilian clothes.

Retrieved items:
<item 1>
John Brown's raid on Harpers Ferry
The Kennedy Farmhouse served as `` barracks , arsenal , supply depot , mess hall , debate club , and home '' . It was very crowded and life there was tedious .
Brown was worried about arousing neighbors ' suspicions . As a result , the raiders had to stay indoors during the daytime , without much to do but study ,
drill , argue politics , discuss religion , and play cards and checkers . Brown 's daughter - in - law Martha served as cook and housekeeper . His daughter
Annie served as lookout . Brown wanted women at the farm , to prevent suspicions of a large all - male group . The raiders went outside at night to drill and
get fresh air . Thunderstorms were welcome since they concealed noise from Brown 's neighbors . Brown did not plan to have a sudden raid and escape to the
mountains . Rather , he intended to use those rifles and pikes he captured at the arsenal , in addition to those he brought along , to arm rebellious slaves
with the aim of striking terror in the slaveholders in Virginia . He believed that on the first night of action , 200 - 500 black slaves would join his line .
</item 1>

<item 2>
John Brown's raid on Harpers Ferry
Contents ( hide ) 1 Brown 's preparation 2 Advance knowledge of raid 3 The raid 3.1 October 16 3.2 October 17 3.3 October 18 3.4 October 19 4 Aftermath 5
Casualties 5.1 John Brown 's raiders 5.2 Others 6 Liberated slaves 7 See also 8 Notes 8.1 Citations 8.2 Bibliography 9 External links Brown 's preparation (
edit ) John Brown rented the Kennedy Farmhouse , with a small cabin nearby , 4 miles ( 6.4 km ) north of Harpers Ferry near the community of Dargan in
Washington County , Maryland , and took up residence under the name Isaac Smith . Brown came with a small group of men minimally trained for military action .
His group included 18 men besides himself ( 13 white men , 5 black men ) . Northern abolitionist groups sent 198 breech - loading . 52 caliber Sharps carbines (
`` Beecher 's Bibles '' ) and 950 pikes ( obtained in late September from Charles Blair of Collinsville Axe Co.
</item 2>

<item 3>
John Brown's raid on Harpers Ferry
During the day , four townspeople were killed , including the mayor . Realizing his escape was cut , Brown took nine of his captives and moved into the smaller
engine house , which would come to be known as John Brown 's Fort . The raiders blocked entry of the windows and doors and traded sporadic gunfire with
surrounding forces . At one point Brown sent out his son , Watson , and Aaron Dwight Stevens with a white flag , but Watson was mortally wounded and Stevens was
shot and captured . The raid was rapidly failing . One of Brown 's men , William H. Leeman , panicked and made an attempt to flee by swimming across the Potomac
River , but he was shot and fatally injured while doing so . During the intermittent shooting , Brown 's other son , Oliver , was also hit ; he died after a
brief period . At around 3 : 00 p.m. a militia company led by Captain E.G. Alburtis arrived by train from Martinsburg , Virginia . Most of the militia members
were Baltimore & Ohio Railroad employees . The militia forced the raiders inside the engine house .
</item 3>

<item 4>
John Brown's raid on Harpers Ferry
2007 On Sunday night , October 16 , 1859 , Brown left four of his men behind as a rear - guard : his son , Owen Brown , Barclay Coppock , and Frank Meriam ; he
led the rest into the town of Harpers Ferry , Virginia . Brown detached a party under John Cook Jr. to capture Colonel Lewis Washington , great - grandnephew of
George Washington , at his nearby Beall - Air estate , some of his slaves , and two relics of George Washington : a sword allegedly presented to Washington by
Frederick the Great and two pistols given by Marquis de Lafayette , which Brown considered talismans . The party carried out its mission and returned via the
Allstadt House , where they took more hostages . Brown 's main party captured several watchmen and townspeople in Harpers Ferry . Brown 's men needed to capture
the weapons and escape before word could be sent to Washington . The raid was going well for Brown 's men . They cut the telegraph wire and seized a Baltimore &
Ohio train passing through . A free black man was the first casualty of the raid .
</item 4>

<item 5>
John Brown's raid on Harpers Ferry
Marines : 1 killed 1 wounded Virginia and Maryland Militia : Unknown 10 killed 7 captured 5 escaped Civilians : 6 killed 9 wounded John Brown 's raid on Harper
's Ferry ( also known as John Brown 's raid or The raid on Harper 's Ferry ) was an effort by armed abolitionist John Brown to initiate an armed slave revolt in
1859 by taking over a United States arsenal at Harpers Ferry , Virginia . Brown 's party of 22 was defeated by a company of U.S. Marines , led by First
Lieutenant Israel Greene . Colonel Robert E. Lee was in overall command of the operation to retake the arsenal . John Brown had originally asked Harriet Tubman
and Frederick Douglass , both of whom he had met in his transformative years as an abolitionist in Springfield , Massachusetts , to join him in his raid , but
Tubman was prevented by illness and Douglass declined , as he believed Brown 's plan would fail .
</item 5>

<item 6>
John Brown's raid on Harpers Ferry
John Brown 's raid on Harpers Ferry - wikipedia John Brown 's raid on Harpers Ferry Jump to : navigation , search John Brown 's Raid on Harper 's Ferry Part of
pre-Civil War conflicts Harper 's Weekly illustration of U.S. Marines attacking John Brown 's `` Fort '' Date October 16 -- 18 , 1859 Location Harpers Ferry , (
West ) Virginia Result U.S. victory Belligerents United States Abolitionist Insurgents Commanders and leaders Robert E. Lee Israel Greene Jeb Stuart John Brown
Strength 88 U.S. Marines Unknown number of Virginia Militia and Maryland Militia 8 white men 12 free black men 1 freed slave 1 fugitive slave Casualties and
losses U.S.
</item 6>

<item 7>
John Brown's raid on Harpers Ferry
, Collinsville , CT ) , in preparation for the raid . The United States Armory was a huge complex of buildings that manufactured small arms for the U.S. Army (
1801 -- 1861 ) , with an Arsenal ( storehouse ) that was thought to contain 100,000 muskets and rifles at the time . An illustration of Brown trying to persuade
abolitionist Frederick Douglass to join him in the raid on Harpers Ferry . Douglass refused . Brown attempted to attract more black recruits . He tried
recruiting Frederick Douglass as a liaison officer to the slaves in a meeting held in a quarry at Chambersburg , Pa . It was at this meeting that ex-slave ``
Emperor '' Shields Green consented to join with John Brown on his attack on the United States Armory , Green stating to Douglass `` I believe I will go with the
old man '' . Douglass declined , indicating to Brown that he believed the raid was a suicide mission . The plan was `` an attack on the federal government ''
that `` would array the whole country against us ... You will never get out alive '' , he warned .
</item 7>

<item 8>
John Brown's raid on Harpers Ferry
A free black man was the first casualty of the raid . Hayward Shepherd , an African - American baggage handler on the train , confronted the raiders ; they shot
and killed him . For some reason , Brown let the train continue , and the conductor alerted the authorities down the line . Brown had been sure that he would
win the support of local slaves in joining the rebellion , but a massive uprising did not occur , because word had not been spread about the uprising , so the
slaves nearby did not know about it . Although the white townspeople soon began to fight back against the raiders , Brown 's men succeeded in capturing the
armory that evening . October 17 ( edit ) Army workers discovered Brown 's men early on the morning of October 17 . Local militia , farmers and shopkeepers
surrounded the armory . When a company of militia captured the bridge across the Potomac River , any route of escape for the raiders was cut off . During the
day , four townspeople were killed , including the mayor .
</item 8>

<item 9>
John Brown's raid on Harpers Ferry
The militia forced the raiders inside the engine house . They broke into the guardroom and freed over two dozen prisoners . Eight militiamen were wounded .
Alburtis said that he could have ended the raid with help from other citizens . By 3 : 30 that afternoon , President James Buchanan ordered a company of U.S.
Marines ( the only government troops in the immediate area ) to march on Harpers Ferry under the command of Brevet Colonel Robert E. Lee , lieutenant colonel of
the 2nd U.S. Cavalry Regiment. Lee had been on leave from his regiment , stationed in Texas , when he was hastily called to lead the detachment and had to
command it while wearing his civilian clothes . October 18 ( edit ) This section needs additional citations for verification . Please help improve this article
by adding citations to reliable sources . Unsourced material may be challenged and removed . ( October 2017 ) ( Learn how and when to remove this template
message ) Lee first offered the role of attacking the engine house to the local militia units on the spot . Both militia commanders declined , and Lee turned to
the Marines .
</item 9>

<item 10>
John Brown's raid on Harpers Ferry
Regardless , one was killed trying to escape across the Potomac River , the other was wounded and later died in the Charles Town jail . ) 9 other civilians were
wounded . Marines Private Luke Quinn ( Killed during the storming of the engine house . ) Buried in Harpers Ferry Catholic Cemetery on Rte. 340 . Private
Matthew Ruppert ( Shot in the face during the storming of the engine house ; survived . ) Liberated slaves ( edit ) Mayor Beckham 's Will Book called for the
liberation of Isaac Gilbert , his wife , and three children upon his death . When Edwin Coppock killed Beckham the five slaves were free . See also ( edit )
List of incidents of civil unrest in the United States Notes ( edit ) Citations ( edit ) ^ Jump up to : `` John Brown 's raid on Harpers Ferry '' . History.com
. Retrieved 1 May 2015 . Jump up ^ In many books the town is called `` Harper 's Ferry . '' For example , `` Col.
</item 10>
=== END ITEM id=82081dc9df9ffb17 ===

=== ITEM id=3f183b74c9b59e7b ===
Gold sentences and facts:
Sentence 0: Network officials , concerned for the show 's future , turned to Tim Russert , the network 's Washington , D.C. , bureau chief . He took over as moderator of Meet the Press on December 8 , 1991 , and remained with the program until his death on June 13 , 2008 , becoming the longest serving moderator in the program 's history .
  Fact 0.0: Network officials were concerned for the show's future.
  Fact 0.1: Network officials turned to Tim Russert.
  Fact 0.2: Tim Russert was the network's Washington, D.C., bureau chief.
  Fact 0.3: Tim Russert took over as moderator of Meet the Press on December 8, 1991.
  Fact 0.4: Tim Russert remained with Meet the Press until his death on June 13, 2008.
  Fact 0.5: Tim Russert became the longest serving moderator in the history of Meet the Press.
Sentence 1: Martha Rountree 1947 -- 1953 Ned Brooks 1953 -- 1965 Lawrence E. Spivak 1966 -- 1975 Bill Monroe 1975 -- 1984 Roger Mudd and Marvin Kalb ( co-moderators ) 1984 -- 1985 Marvin Kalb 1985 -- 1987 Chris Wallace 1987 -- 1988 Garrick Utley 1989 -- 1991 Tim Russert 1991 -- 2008 Tom Brokaw 2008 David Gregory 2008 -- 2014 Chuck Todd 2014 -- present
  Fact 1.0: Martha Rountree was moderator from 1947 to 1953.
  Fact 1.1: Ned Brooks was moderator from 1953 to 1965.
  Fact 1.2: Lawrence E. Spivak was moderator from 1966 to 1975.
  Fact 1.3: Bill Monroe was moderator from 1975 to 1984.
  Fact 1.4: Roger Mudd and Marvin Kalb were co-moderators from 1984 to 1985.
  Fact 1.5: Marvin Kalb was moderator from 1985 to 1987.
  Fact 1.6: Chris Wallace was moderator from 1987 to 1988.
  Fact 1.7: Garrick Utley was moderator from 1989 to 1991.
  Fact 1.8: Tim Russert was moderator from 1991 to 2008.
  Fact 1.9: Tom Brokaw was moderator in 2008.
  Fact 1.10: David Gregory was moderator from 2008 to 2014.
  Fact 1.11: Chuck Todd has been moderator since 2014.

Retrieved items:
<item 1>
Meet the Press
It originates from NBC 's bureau in Washington , D.C. The longevity of Meet the Press can be illustrated in consideration of the fact that the program debuted
during what was only the second official `` network television season '' for American television . One historical landmark of the program is that it was the
first on which a sitting U.S. President , Gerald Ford , appeared on a live television network news program , which occurred on the November 9 , 1975 broadcast .
The program has been hosted by 12 different moderators to date , beginning with creator Martha Rountree . The show 's moderator since 2014 is Chuck Todd , who
also serves as political director for NBC News . Currently , the hour - long program airs in most markets on Sundays at 9 : 00 a.m. live in the Eastern Time
Zone and on tape delay elsewhere . Meet the Press is also occasionally pre-empted due to network coverage of sports events held outside of the U.S. The program
is also rebroadcast on Sundays at 2 : 00 p.m. , and Mondays at 2 : 00 a.m. and sometimes 4 : 00 a.m.
</item 1>

<item 2>
Meet the Press
On May 2 , 2010 , Meet the Press became the last NBC News program to convert to high definition , and unveiled a new set consisting of large video screens
mostly used to display Washington scenery , satellite interview subjects and moderator and subject talking points , along with graphics produced for the format
. Moderators ( edit ) The following is the list of moderators for Meet the Press : Martha Rountree 1947 -- 1953 Ned Brooks 1953 -- 1965 Lawrence E. Spivak 1966
-- 1975 Bill Monroe 1975 -- 1984 Roger Mudd and Marvin Kalb ( co-moderators ) 1984 -- 1985 Marvin Kalb 1985 -- 1987 Chris Wallace 1987 -- 1988 Garrick Utley
1989 -- 1991 Tim Russert 1991 -- 2008 Tom Brokaw 2008 David Gregory 2008 -- 2014 Chuck Todd 2014 -- present On - remote broadcasts ( outside D.C. studios ) (
edit ) Date Location Event / Reason July 17 , 1988 Atlanta , Georgia 1988 Democratic National Convention August 14 , 1988 New Orleans ,
</item 2>

<item 3>
Meet the Press
studios ) 3.2 Notable guests and events 4 Distribution 5 See also 6 References and footnotes 7 External links Format ( edit ) The program 's format consists of
an extended one - on - one interview with the host , and is sometimes followed by a roundtable discussion or one - on - two interview with figures in
adversarial positions , either Congressional members from opposite sides of the aisle or political commentators . Originally a half - hour program for most of
its history , the show expanded to 60 minutes starting with the September 20 , 1992 , broadcast . The program also features in - depth examinations of facts
behind political and general news stories ( particularly as part of a segment called the `` Data Download '' , introduced after Chuck Todd assumed duties as
moderator , which is conducted on a touchscreen within the main set . History ( edit ) Meet the Press set , November 1975 . On this broadcast , a sitting
American president ( Gerald Ford ) was , for the first time , a guest on a live television network news program .
</item 3>

<item 4>
Graph query row:
person: Tim Russert
work: Meet the Press
relationship: RELATED_TO
role: moderator
tenure: None
period: 1991 -- 2008
term: None
context: None
fact: None
</item 4>

<item 5>
Meet the Press
The following week on August 17 , he moderated the entire broadcast . It was also reported on December 1 , 2008 , that the December 7 broadcast would be Brokaw
's last , with Gregory becoming the new permanent host the following Sunday . Under Gregory ( edit ) David Gregory began his tenure as moderator on December 14
, 2008 . Four days after Gregory 's first regular broadcast , on December 18 , NBC News political director Chuck Todd was named contributing editor of Meet the
Press . Throughout Gregory 's tenure as moderator , Meet the Press experienced significant ratings declines . In the final three months of 2013 , the program
placed third among the Sunday morning talk shows in total viewership , behind CBS 's Face the Nation and ABC 's This Week for the first time since 1992 ; it
also experienced the lowest ratings in the show 's entire history among the key 25 - to - 54 age viewing demographic during this period . NBC management became
uncertain as to the future direction of the program .
</item 5>

<item 6>
Meet the Press
Russert interviews General Peter Pace in 2006 . Network officials , concerned for the show 's future , turned to Tim Russert , the network 's Washington , D.C.
, bureau chief . He took over as moderator of Meet the Press on December 8 , 1991 , and remained with the program until his death on June 13 , 2008 , becoming
the longest serving moderator in the program 's history . Under Russert , the program was expanded to one hour and became less of a televised press conference ,
focusing more on Russert 's questions and comments ; Russert also engaged in longer in - depth interviews and hosted panels of experts to discuss the topics
featured in that week 's broadcast . Russert signed off each edition by saying , `` That 's all for today . We 'll be back next week . If it 's Sunday , it 's
Meet the Press . '' During the professional football season , Russert , a native of Buffalo , New York , and an avid fan of the Buffalo Bills , sometimes added
, `` Go Bills ! , '' and occasionally would ask panelists , `` How ' bout those Sabres ? ''
</item 6>

<item 7>
Meet the Press
The next week , Bill Monroe , previously a weekly panelist like Spivak had been years before , took over as moderator and stayed until June 2 , 1984 . For the
next seven and a half years , the program then went through a series of hosts as it struggled in the ratings against ABC 's This Week with David Brinkley .
Roger Mudd and Marvin Kalb , as co-moderators , followed Monroe for a year , followed by Chris Wallace ( who would later to go on to a much longer run as host
of the rival program Fox News Sunday ) from 1987 to 1988 . Garrick Utley , then hosting Weekend Today , concurrently hosted Meet the Press from 1989 through
December 1 , 1991 . All this occurred despite the increasing ratings of NBC News ' other programs ( and those of the network generally ) during that period .
The program originally aired at noon Eastern Time every Sunday , before moving to a 9 : 00 a.m. slot by the early 1990s . Under Russert ( edit ) Meet the Press
logo used from September 10 , 1995 to June 2008 .
</item 7>

<item 8>
Meet the Press
This section , in the next two paragraphs , needs additional citations for verification . Please help improve this article by adding citations to reliable
sources . Unsourced material may be challenged and removed . ( September 2014 ) ( Learn how and when to remove this template message ) Meet the Press was
originally presented as a 30 - minute press conference with a single guest and a panel of questioners . Its first guest was James Farley , who served as
Postmaster General , Democratic National Committee chairman and campaign manager to Franklin Delano Roosevelt under the first two terms of the New Deal
Administration . Creator Rountree served as its first host , the program 's only female moderator to date . She stepped down on November 1 , 1953 , and was
succeeded by Ned Brooks , who remained as moderator until his retirement on December 26 , 1965 . Spivak became the moderator on January 1 , 1966 , moving up
from his role as a permanent panelist . He retired on November 9 , 1975 , on a special one - hour edition that featured , for the first time , a sitting
president , Gerald Ford , as guest .
</item 8>

<item 9>
Meet the Press
Camera setup Multi-camera Running time 30 minutes ( 1947 -- 1992 ) 52 minutes ( 1992 -- present ) Production company ( s ) NBC News Productions Release Original
network NBC Picture format 480i ( 4 : 3 SDTV ) ( 1947 -- 2010 ) 720p ( 4 : 3 SDTV ) ( 1996 -- 2010 ) 1080i ( 16 : 9 HDTV ) ( 2010 -- present ) Original release
November 6 , 1947 ( 1947 - 11 - 06 ) -- present External links Website www.nbcnews.com/meet-the-press Meet the Press is a weekly American television news /
interview program broadcast on NBC . It is the longest - running program in television history , though the current format bears little resemblance to the debut
episode on November 6 , 1947 . Meet the Press specializes in interviews with leaders in Washington , D.C. , across the country and even the world on issues of
politics , economics , foreign policy and other public affairs , along with panel discussions that provide opinions and analysis . It originates from NBC 's
bureau in Washington , D.C.
</item 9>

<item 10>
Meet the Press
Meet the Press - Wikipedia Meet the Press Jump to : navigation , search This article is about the American public affairs program . For the Australian program ,
see Meet the Press ( Australian TV program ) . Meet the Press Genre Public affairs / news analysis program Created by Martha Rountree Lawrence E. Spivak
Directed by Rob Melick Presented by Chuck Todd ( for past moderators , see section ) Narrated by Fred Facey Bert Pence Dennis Haysbert Theme music composer John
Williams Opening theme `` The Pulse of Events '' ( fourth part of The Mission ) Country of origin United States Original language ( s ) English No. of seasons
70 No. of episodes 3,600 + ( weekly ) ( as of 2017 ) Production Executive producer ( s ) John Reiss Location ( s ) NBC News Washington Bureau , Washington ,
D.C.
</item 10>
=== END ITEM id=3f183b74c9b59e7b ===

=== ITEM id=160076d64f138281 ===
Gold sentences and facts:
Sentence 0: A justice of the peace ( JP ) is a judicial officer , of a lower or puisne court , elected or appointed by means of a commission ( letters patent ) to keep the peace . In past centuries the term commissioner of the peace was often used with the same meaning . Depending on the jurisdiction , such justices dispense summary justice or merely deal with local administrative applications in common law jurisdictions . Justices of the peace are appointed or elected from the citizens of the jurisdiction in which they serve , and are ( or were ) usually not required to have any formal legal education in order to qualify for the office . Some jurisdictions have varying forms of training for JPs .
  Fact 0.0: A justice of the peace (JP) is a judicial officer.
  Fact 0.1: A justice of the peace is a judicial officer of a lower or puisne court.
  Fact 0.2: A justice of the peace is elected or appointed by means of a commission (letters patent).
  Fact 0.3: A justice of the peace is elected or appointed to keep the peace.
  Fact 0.4: In past centuries the term commissioner of the peace was often used with the same meaning as justice of the peace.
  Fact 0.5: Depending on the jurisdiction, justices of the peace dispense summary justice.
  Fact 0.6: Depending on the jurisdiction, justices of the peace merely deal with local administrative applications in common law jurisdictions.
  Fact 0.7: Justices of the peace are appointed or elected from the citizens of the jurisdiction in which they serve.
  Fact 0.8: Justices of the peace are (or were) usually not required to have any formal legal education in order to qualify for the office.
  Fact 0.9: Some jurisdictions have varying forms of training for justices of the peace.

Retrieved items:
<item 1>
Justice of the peace
These courts were introduced in 2009 as a replacement for the district courts ( established in 1975 ) , which in turn replaced burgh police courts . Justices
sit alone or in threes with a qualified legal assessor as convener or clerk of court . They handle many cases of breaches of the peace , drunkenness , minor
assaults , petty theft , and offences under the Civic Government ( Scotland ) Act 1982 . The maximum sentencing power of a justice of the peace is 60 days
imprisonment , or a fine up to £ 2,500 , or both , and the ability to disqualify drivers . In 2006 , the Scottish Government announced its intention to unify
the management of the sheriff and district courts in Scotland , but retaining lay justices , as part of its initiative to create a unified judiciary under the
Lord President . Following the passage of the Criminal Proceedings etc . ( Reform ) ( Scotland ) Act 2007 the justice of the peace courts were implemented on a
sheriffdom - by - sheriffdom basis . In Glasgow , the volume of business required the employment of three solicitors as `` stipendiary magistrates '' who sat in
place of the lay justices .
</item 1>

<item 2>
Justice of the peace
( PLD 2016 Supreme Court 581 ) It is pertinent to note however , as many academics have pointed out , that there is great utility in the appointment of such
justices especially in rural areas where enmity between rival groups can lead to the inability of registration of cognizable offences and biased judicial
proceedings . Singapore ( edit ) A justice of the peace in Singapore derives his powers from statute law . He is appointed by the President of the Republic of
Singapore , under the provisions of section 11 ( l ) of the Subordinate Courts Act ( Cap. 321 ) . The President may revoke the appointment of any justice of the
peace . A newly appointed justice of the peace is required by section 17 of the Subordinate Courts Act , to take the oath of office and allegiance as set out in
the schedule to the Subordinate Courts Act , before exercising the functions of his office . In their role , JPs serve as visiting justices of Singapore prisons
, mediators or referees in the subordinate courts . They may also solemnise marriages in the Registry of Marriages or perform the duties of the magistrate
conferred on them by any written law .
</item 2>

<item 3>
Justice of the peace
Candidates must successfully complete a three - day training course run by the Department of Justice . Bail justices , who are also volunteers , also have some
limited powers under federal legislation , including the power to conduct interstate extradition hearings and extending question time for federal police . Bail
Justices may use the post-nominals BJ after their names . New South Wales ( edit ) The most common functions performed by a justice of the peace in New South
Wales are to witness the signing of a statutory declaration , witness the signing of an affidavit and certify that a copy of an original document is a true copy
. JPs are appointed by the Governor of New South Wales for five - year terms . They are volunteers , who come from all walks of life and all sections of the
community . JPs are people who are trusted to be honest , careful and impartial when performing the functions of a JP . They must not charge a fee or accept a
gift for providing JP services , tell people what to write in a statutory declaration or affidavit or write it for them or give them legal advice . Ways to find
a JP in New South Wales include : 1 . Search the JP Public Register .
</item 3>

<item 4>
Justice of the peace
Justice of the peace - wikipedia Justice of the peace Jump to : navigation , search This article needs additional citations for verification . Please help
improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( January 2017 ) ( Learn how and when to
remove this template message ) Taos County , New Mexico . Justice of the Peace Montoyo hears a minor case . ( 1941 ) A justice of the peace ( JP ) is a judicial
officer , of a lower or puisne court , elected or appointed by means of a commission ( letters patent ) to keep the peace . In past centuries the term
commissioner of the peace was often used with the same meaning . Depending on the jurisdiction , such justices dispense summary justice or merely deal with
local administrative applications in common law jurisdictions . Justices of the peace are appointed or elected from the citizens of the jurisdiction in which
they serve , and are ( or were ) usually not required to have any formal legal education in order to qualify for the office . Some jurisdictions have varying
forms of training for JPs . Contents ( hide ) 1 History 2 Modern use 2.1 Australia 2.1.
</item 4>

<item 5>
Justice of the peace
Their judicial powers were replaced by full - time , legally qualified district justices ( now called district judges ) and their quasi-judicial powers by lay
Peace Commissioners . Peace commissioners may sign statutory declarations , and may issue summons and search warrants to the Garda Síochána ( Irish police ) . A
Peace Commissioner can not sign an Affidavit , in Ireland that function is performed by a Commissioner for Oaths . Jamaica ( edit ) A justice of the peace ( JP
) , according to the Ministry of Justice , is a person of unquestionable integrity who seeks to promote and protect the rights of the individual and helps to
provide justice to persons in a particular community . Additionally , the JP serves as a justice in petty court sessions , attends juvenile court sessions ,
issues summonses , considers applications for bail , explains and signs legal documents , sits on licensing panels , and gives counsel / advice . Any Jamaican
citizen that can speak and write English is eligible to become a JP . Any club / organisation / citizen can recommend someone to become JP for a community . JPs
are chosen under the Governor - General 's discretion .
</item 5>

<item 6>
Justice of the peace
JPs are chosen under the Governor - General 's discretion . Malaysia ( edit ) In Malaysia , justices of the peace have largely been replaced in magistrates '
courts by legally qualified ( first - class ) stipendiary magistrates . However , state governments continue to appoint justices of the peace as honours . In
2004 , some associations of justices of the peace pressed the federal government to allow justices of the peace to sit as second - class magistrates in order to
reduce the backlog of cases in the courts . New Zealand ( edit ) A Justice of the Peace in New Zealand is someone of good stature in the community who is
authorized to witness and sign statutory declarations and affidavits as well as certify documents . They may also perform citizenship ceremonies and act as a
Visiting Justice in prisons the function of which is to record a prisoner 's grievance and decide on its validity . They have certain powers to issue search
warrants , and ( in conjunction with another justice of the peace ) may try minor criminal trials in the district court and exercise powers to remand defendants
in custody , grant bail , and adjourn court hearings .
</item 6>

<item 7>
Justice of the peace
, dominance of the land - owning gentry ) . For example , historian Tim Blanning notes that while in Britain the royal prerogative was decisively curbed by the
Bill of Rights 1689 , in practice the central government in London had a greater ability to get its policies implemented in the rural outlying regions than
could contemporary absolute monarchies such as France -- a paradox due especially to JPs belonging to the same social class as the Members of Parliament and
thus having a direct interest in getting laws actually enforced and implemented on the ground . Being an unpaid office , undertaken voluntarily and sometimes
more for the sake of renown or to confirm the justice 's standing within the community , the justice was typically a member of the gentry . The justices of the
peace conducted arraignments in all criminal cases , and tried misdemeanours and infractions of local ordinances and bylaws . Towns and boroughs with enough
burdensome judicial business that could not find volunteers for the unpaid role of justice of the peace had to petition the Crown for authority to hire a paid
stipendiary magistrate . The Municipal Corporations Act 1835 stripped the power to appoint normal JPs from those municipal corporations that had it .
</item 7>

<item 8>
Justice of the peace
United states ( edit ) In some US states , the justice of the peace is a judge of a court of limited jurisdiction , a magistrate , or a quasi-judicial official
with certain statutory or common law magisterial powers . Some states have special qualifications or unique features for the office . The justice of the peace
typically presides over a court that hears misdemeanor cases , traffic violations , and other petty criminal infractions . The justice of the peace may also
have authority over cases involving small debts , landlord and tenant disputes , or other small claims court proceedings . Proceedings before justices of the
peace are often faster and less formal than the proceedings in other courts . In some jurisdictions a party convicted or found liable before a justice of the
peace may have the right to a trial de novo before the judge of a higher court rather than an appeal strictly considered . A justice of the peace also performs
civil marriages . Arizona ( edit ) A justice of the peace has the same jurisdiction as a municipal magistrate with respect to traffic and misdemeanor cases and
restraining orders , though over cases whose affairs are not contained within the confines of a single municipality .
</item 8>

<item 9>
Justice of the peace
Administrative justice of the peace are court officers appointed by the Minister of Justice , and perform duties such as receiving criminal informations and
issuing warrants . Presiding justice of the peace are appointed by commission under the Great Seal , and can try some criminal matters and issue warrants . They
are appointed from advocates of at least ten years ' standing and serve full - time until the age of 70 . Yukon ( edit ) In Yukon , justices of the peace are
lay officers of the court . They sit in the Justice of the Peace Court , which is part of the Territorial Court of Yukon . Hong Kong ( edit ) In Hong Kong , the
historical functions of justices of the peace have been replaced by full - time , legally qualified magistrates . Nowadays , justices of the peace are
essentially titles of honour given by the Government to community leaders , and to certain officials while they are in their terms of offices . They have no
judicial functions , and their main duties include visiting prisons , institutions for young offenders and drug addicts , psychiatric hospitals , remand homes ,
places of refuge , reception and detention centres , administering statutory declarations , and serving as members of advisory panels .
</item 9>

<item 10>
Justice of the peace
JPs , who are also volunteers , are selected through an extensive interview , written exam and practical testing . They are recommended by the state attorney -
general and appointed by the governor - in - council , and it is their job to authorise and witness statutory declarations and affidavits within the state of
Victoria . There are currently around 4,800 JPs serving in all areas of the state . Bail justices ( edit ) The role of a bail justice is to hear bail
applications ( under the Bail Act 1977 ) and to hear applications for interim accommodation orders for children ( under the Children and Young Persons Act 1989
) within Victoria . Bail justices , once appointed , may remain in their role until they turn 70 years of age ( although they must be under 65 at the time of
their appointment ) . They are often required to attend call outs and rule on bail applications or protection applications for children in danger on weekends
and late at night when the courts are closed , but they can also witness Victorian statutory declarations and affidavits . Candidates must successfully complete
a three - day training course run by the Department of Justice .
</item 10>
=== END ITEM id=160076d64f138281 ===

=== ITEM id=c3266d53bc9563e9 ===
Gold sentences and facts:
Sentence 0: This was the city of Chicago 's first professional sports championship since the Chicago Fire won MLS Cup ' 98 ( which came four months after the Chicago Bulls ' sixth NBA championship that year ) . The next major Chicago sports championship came in 2010 , when the NHL 's Chicago Blackhawks ended a 49 - year Stanley Cup title drought . With the Chicago Bears ' win in Super Bowl XX and the Chicago Cubs ' own World Series championship in 2016 , all Chicago sports teams have won at least one major championship since 1985 . Meanwhile , the Astros themselves made it back to the World Series in 2017 , but this time as an AL team , where they defeated the Los Angeles Dodgers in seven games , resulting in Houston 's first professional sports championship since the 2006 -- 07 Houston Dynamo won their back - to - back MLS Championships .
  Fact 0.0: The championship in this context was the first professional sports championship for the city of Chicago since the Chicago Fire won MLS Cup '98.
  Fact 0.1: The Chicago Fire won MLS Cup '98.
  Fact 0.2: The Chicago Fire's MLS Cup '98 win came four months after the Chicago Bulls' sixth NBA championship that year.
  Fact 0.3: The next major Chicago sports championship came in 2010.
  Fact 0.4: In 2010, the NHL's Chicago Blackhawks ended a 49-year Stanley Cup title drought.
  Fact 0.5: The Chicago Bears won Super Bowl XX.
  Fact 0.6: The Chicago Cubs won the World Series in 2016.
  Fact 0.7: All Chicago sports teams have won at least one major championship since 1985.
  Fact 0.8: The Astros made it back to the World Series in 2017.
  Fact 0.9: The Astros were in the 2017 World Series as an American League team.
  Fact 0.10: In 2017, the Astros defeated the Los Angeles Dodgers in seven games in the World Series.
  Fact 0.11: The Astros' 2017 World Series win resulted in Houston's first professional sports championship since the 2006-07 Houston Dynamo won their back-to-back MLS Championships.
  Fact 0.12: The Houston Dynamo won back-to-back MLS Championships in 2006-07.
Sentence 1: Neither team advanced to the post-season in 2006 , but the 2006 World Series again featured teams from the American League Central and National League Central divisions , this time represented by the Detroit Tigers and the St. Louis Cardinals , respectively . The Cardinals won the World Series in five games , in which manager Tony La Russa became the second manager to win the World Series in both American and National leagues , previously managing the Oakland Athletics to the 1989 World Series championship . Both the White Sox and the Astros were in the Wild Card race until the final weeks of the season , with the White Sox finishing with 90 wins , the Astros with 82 wins . The White Sox made their first post-2005 playoff appearance in 2008 , while the Astros would not return to the postseason until 2015 , their third season as an American League team and would not return to the World Series until 2017 , their fifth season as an American League team .
  Fact 1.0: Neither team advanced to the post-season in 2006.
  Fact 1.1: The 2006 World Series again featured teams from the American League Central and National League Central divisions.
  Fact 1.2: The Detroit Tigers and the St. Louis Cardinals represented the American League Central and the National League Central, respectively, in the 2006 World Series.
  Fact 1.3: The St. Louis Cardinals won the World Series in five games.
  Fact 1.4: In the 2006 World Series, manager Tony La Russa became the second manager to win the World Series in both the American and National leagues.
  Fact 1.5: Tony La Russa had previously managed the Oakland Athletics to the 1989 World Series championship.
  Fact 1.6: Both the White Sox and the Astros were in the Wild Card race until the final weeks of the season.
  Fact 1.7: The White Sox finished the season with 90 wins.
  Fact 1.8: The Astros finished the season with 82 wins.
  Fact 1.9: The White Sox made their first post-2005 playoff appearance in 2008.
  Fact 1.10: The Astros would not return to the postseason until 2015, their third season as an American League team.
  Fact 1.11: The Astros would not return to the World Series until 2017, their fifth season as an American League team.

Retrieved items:
<item 1>
2017 World Series
In the bottom of the inning , Yasiel Puig hit a home run off of Ken Giles and Enrique Hernández drove in Logan Forsythe to tie the game , with the latter being
the Dodgers ' first run that was not driven in by a home run . In the next inning , George Springer hit a two - run home run for the Astros off of Brandon
McCarthy to retake the lead . In the bottom of the 11th inning , Charlie Culberson homered off of Chris Devenski , who later struck out Puig to end the game .
This was the first ever World Series game in which a team hit home runs in the ninth , tenth and eleventh inning . The teams set a new record for combined home
runs in a single World Series game with eight and this was the first time in MLB history , regular season or postseason , that five home runs were hit in extra
innings . The Astros won their first World Series game in franchise history as they had been swept in their previous appearance in 2005 . Game 3 ( edit ) Lance
McCullers Jr. earned the win in Game 3 .
</item 1>

<item 2>
Houston Astros
-- 2014 ) Tom Lawless ( 2014 ) A.J. Hinch ( 2015 -- ) Houston Astros retired numbers 5 Jeff Bagwell 7 Craig Biggio 24 Jimmy Wynn 25 José Cruz 32 Jim Umbricht 33
Mike Scott 34 Nolan Ryan 40 Don Wilson 49 Larry Dierker Houston Astros 2017 World Series champions 1 Carlos Correa 2 Alex Bregman 3 Cameron Maybin 4 George
Springer ( World Series MVP ) 9 Marwin González 10 Yuli Gurriel 11 Evan Gattis 15 Carlos Beltrán 16 Brian McCann 21 Derek Fisher 22 Josh Reddick 27 José Altuve
( AL MVP ) 30 Juan Centeno 31 Collin McHugh 35 Justin Verlander ( ALCS MVP ) 36 Will Harris 41 Brad Peacock 43 Lance McCullers Jr . 44 Luke Gregerson 46
Francisco Liriano 47 Chris Devenski 50 Charlie Morton 53 Ken Giles 59 Joe Musgrove 60 Dallas Keuchel Manager 14 A.J.
</item 2>

<item 3>
Houston Astros
Cellular Field in Chicago and the final two in Houston . Game 3 marked the first World Series game held in the state of Texas , and was the longest game in
World Series history , lasting 5 hours and 41 minutes . This World Series was marked by a controversy involving the Minute Maid Park roof . MLB & Commissioner
Bud Selig insisted that the Astros must play with the roof open which mitigated the intensity and enthusiasm of the cheering Astros fans . 2006 -- 2010 : the
decline ( edit ) In the 2006 offseason , the team signed Preston Wilson and moved Berkman to first base , ending the long tenure of Jeff Bagwell . The Astros
renewed the contract with Clemens and traded two minor league prospects to the Tampa Bay Devil Rays for left - handed hitter Aubrey Huff . By August , Preston
Wilson complained about his playing time after the return of Luke Scott from AAA Round Rock . The Astros released Wilson and he was signed by St. Louis . A
dramatic season end included wins in 10 of their last 12 games , but the Astros missed a playoff appearance when they lost the final game of the season to the
Atlanta Braves .
</item 3>

<item 4>
2017 World Series
Forsythe drove in Cody Bellinger to tie the game in the top of the seventh . Morton struck out seven and only allowed three hits and one run in 6 ​ ⁄ innings .
Bellinger then drove in the go - ahead run with a double in the top of the ninth off of Giles and the Dodgers added four more runs on a sacrifice fly by Austin
Barnes and a three - run homer by Pederson . Bregman homered off of Jansen in the bottom of the ninth but the Dodgers managed to even up the series . The Astros
had two hits in the game ; both were home runs . This was the first game in World Series history where both starting pitchers allowed four or fewer baserunners
. After Game 4 , both teams ' pitching coaches , Rick Honeycutt for the Dodgers and Brent Strom for the Astros , commented on how the baseball being used for
the World Series is slicker than the baseball used during the regular season . Pitchers on both teams noted that this difference has made it more difficult for
them to throw their sliders . Game 5 ( edit ) Alex Bregman drove in the game - winning run in Game 5 .
</item 4>

<item 5>
Houston Astros
The contributions of the other starters -- Brandon Backe ( 10 -- 8 , 4.76 ) and rookie starters Ezequiel Astacio ( 3 -- 6 , 5.67 ) and Wandy Rodríguez ( 10 --
10 , 5.53 ) -- were less remarkable , but enough to push the Astros into position for a playoff run . The Astros won a wild card berth on the final day of the
regular season , becoming the first team since the world champion 1914 Boston Braves to qualify for the postseason after being 15 games under . 500 . The Astros
won the National League Division Series against the Atlanta Braves , 3 - 1 , with a game four that set postseason records for most innings ( 18 ) , most players
used by a single team ( 23 ) , and longest game time ( 5 hours and 50 minutes ) . Trailing by a score of 6 - 1 , Lance Berkman hit an eighth inning grand slam
to narrow the score to 6 -- 5 . In the bottom of the ninth , catcher Brad Ausmus , hit a game tying home run that allowed the game to continue in extra innings
.
</item 5>

<item 6>
2005 World Series
Pierzynski hit a two - out double to Tal 's Hill , driving in two runs , scoring Iguchi and Dye giving the Sox the lead . The Astros rallied in the last of the
eighth with two outs when Lane 's double scored Ensberg with the tying run after back - to - back walks by Ensberg and Mike Lamb , giving Dustin Hermanson a
blown save . Houston tried to rally to win in the ninth , but stranded Chris Burke at third , after he had walked , reached second on an error and stolen third
. The Astros tried again in the 10th as well as in the 11th , but failed each time . In the top of the 14th , after the Sox hit into a spectacular double play
started by Ensberg , Geoff Blum ( a former Astro and the Astros ' television color analyst as of 2015 ) , who had entered the game in the 13th , homered to
right with two outs off Ezequiel Astacio . Infield singles by Rowand and Crede were followed by walks to Uribe and Chris Widger .
</item 6>

<item 7>
2005 World Series
With the Astros on the verge of clinching their first ever National League pennant in Game 5 , Albert Pujols hit a mammoth three - run home run off Brad Lidge
in the top of the ninth inning to take the lead , and subsequently stave off elimination . However , behind NLCS MVP Roy Oswalt , the Astros were able to defeat
the Cards 5 -- 1 in Game 6 and earned a trip to the World Series . This was the Astros ' first World Series appearance in franchise history . Matchups ( edit )
Game 1 ( edit ) Saturday , October 22 , 2005 7 : 05 pm ( CDT ) at U.S.
</item 7>

<item 8>
2005 World Series
The Astros were attempting to become the fourth consecutive wild card team to win the Series , following the Anaheim Angels ( 2002 ) , Florida Marlins ( 2003 )
and Boston Red Sox ( 2004 ) . Both teams were attempting to overcome decades of disappointment , with a combined 132 years between the two teams without a title
. The Astros were making their first Series appearance in 44 years of play , while the White Sox had waited exactly twice as long for a title , having last won
the Series in 1917 , and had not been in the Series since 1959 , three years before the Astros ' inaugural season . Like the 1982 World Series between the St.
Louis Cardinals and the Milwaukee Brewers , the 2005 World Series is one of only two World Series in the modern era ( 1903 -- present ) with no possibility for
a rematch between the two opponents , because the Astros moved to the American League in 2013 . However , the Brewers did meet the Cardinals in the 2011
National League Championship Series .
</item 8>

<item 9>
2017 World Series
October 25 , 2017 , 5 : 17 pm ( PDT ) at Dodger Stadium in Los Angeles , California , 93 ° F ( 34 ° C ) , clear Team 5 6 7 8 9 10 11 Houston 0 0 0 0 0 0 7 14
Los Angeles 0 0 0 0 0 0 0 6 5 0 WP : Chris Devenski ( 1 -- 0 ) LP : Brandon McCarthy ( 0 -- 1 ) Home runs : HOU : Marwin González ( 1 ) , José Altuve ( 1 ) ,
Carlos Correa ( 1 ) , George Springer ( 1 ) LAD : Joc Pederson ( 1 ) , Corey Seager ( 1 ) , Yasiel Puig ( 1 ) , Charlie Culberson ( 1 ) Attendance : 54,293 The
starting pitchers for Game 2 were Rich Hill for the Dodgers and Justin Verlander for the Astros . The Astros scored first when Bregman drove in Josh Reddick
with a hit in the third inning .
</item 9>

<item 10>
2005 World Series
2016 2017 2018 Houston Astros Formerly the Houston Colt . 45s Based in Houston , Texas Franchise Expansion Draft History Seasons Records No - hitters Awards
Players Managers Owners and executives First - round draft picks Opening Day starting pitchers All articles Ballparks Colt Stadium Houston Astrodome Minute Maid
Park Spring training : Geronimo Park Cocoa Expo Stadium Osceola County Stadium The Ballpark of the Palm Beaches Lore 1980 NL West tie - breaker game Killer B 's
2015 AL Wild Card Game Houston Strong Culture AstroTurf Orbit Crawford Boxes Tal 's Hill Continental League Houston Buffs Houston College Classic Ball Four (
Jim Bouton book ) The Bad News Bears in Breaking Training Murder at the World Series Night Game `` The Hot Tub '' ( Seinfeld episode ) Boyhood `` Holy Toledo !
'' Sunshine Kids Darryl Kile Award Key personnel Owner : Jim Crane President : Reid Ryan General Manager : Jeff Luhnow Manager : A.J.
</item 10>
=== END ITEM id=c3266d53bc9563e9 ===
