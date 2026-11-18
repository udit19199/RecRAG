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

=== ITEM id=e5ce0c53c4391514 ===
Gold sentences and facts:
Sentence 0: Subjects are all taken on the same level - there is no higher or standard grade as in the past . There are three pass levels , each with different pass requirements. First is a pass with Higher certificate , one must obtain 40 % in home language , 40 % in two other subjects and must obtain 30 % in three other subjects . A learner who passed matric with higher certificate level , can not enroll for a University Degree nor a Diploma at an institute of higher learning . The mean mark in any subject is usually about 55 . Only a small proportion of candidates score an ' A ' in any subject ( from as little as 2 % to a maximum of about 10 % in subjects taken by highly select groups . ) A further 8 -- 15 % are likely to gain a ' B ' and about 20 -- 25 % achieve a ' C ' grade . The National Senior Certificate is a group certificate and records an aggregate mark .
  Fact 0.0: All subjects are taken on the same level, and there is no higher or standard grade as in the past.
  Fact 0.1: There are three pass levels, each with different pass requirements.
  Fact 0.2: The first pass level is a pass with Higher certificate.
  Fact 0.3: To pass with Higher certificate, a learner must obtain 40% in home language, 40% in two other subjects, and 30% in three other subjects.
  Fact 0.4: A learner who passed matric with Higher certificate level cannot enroll for a University Degree.
  Fact 0.5: A learner who passed matric with Higher certificate level cannot enroll for a Diploma at an institute of higher learning.
  Fact 0.6: The mean mark in any subject is usually about 55.
  Fact 0.7: Only a small proportion of candidates score an 'A' in any subject, from as little as 2% to a maximum of about 10% in subjects taken by highly select groups.
  Fact 0.8: A further 8 to 15% of candidates are likely to gain a 'B' in a subject.
  Fact 0.9: About 20 to 25% of candidates achieve a 'C' grade.
  Fact 0.10: The National Senior Certificate is a group certificate.
  Fact 0.11: The National Senior Certificate records an aggregate mark.

Retrieved items:
<item 1>
National Senior Certificate
V Group a : fundamentals ( compulsory ) ( edit ) Learners must study 4 compulsory subjects , Two official languages of South Africa must be studied : Afrikaans
English Ndebele Northern Sotho Sesotho Siswati Tsonga Tswana Venda Xhosa Zulu Mathematics or Mathematical Literacy Life Orientation Group B : electives ( edit )
At least 3 subjects from the following : Accounting Agricultural Management Practices Agricultural Sciences Agricultural Technology Business Studies Civil
Technology Computer Applications Technology ( previously Computer Studies Standard Grade - Literacy ) Consumer Studies Dance Studies Design Dramatic Arts
Economics Electrical Technology Engineering Graphics & Design Geography History Hospitality Studies ( previously Home Economics ) Information Technology (
previously Computer Studies Higher Grade - Programming ) Life Sciences ( previously Biology ) Mechanical Technology Music Physical Science Religion Studies
Second Additional Language Third Additional Language Tourism Visual Arts Life Orientation ( edit ) Life Orientation ( colloquially abbreviated as `` LO '' ) has
been introduced into the senior high school phase as an examination subject . Life Orientation is a broad - learning subject that covers non-academic skills
needed in everyday life .
</item 1>

<item 2>
National Senior Certificate
The IEB operates on a national level catering primarily for independent schools . Contents ( hide ) 1 Subject requirements 1.1 Group A : Fundamentals (
Compulsory ) 1.2 Group B : Electives 2 Life Orientation 3 Assessment 4 Level system ( 1 to 7 ) 5 The 5 % language compensation 6 University entrance 7 See also
8 References 9 External links Subject requirements. ( edit ) Learners study at least 7 subjects - 4 compulsory and at least 3 electives . All subjects are
written on one grade only and are no longer written on Higher or Standard Grade . Not all schools offer the full range of Elective subjects listed here . Each
school may offer subjects specific to its academic orientation . For example , Agriculture Schools offer the agriculture - orientated subjects whereas technical
Schools offer the practical and mechanical - orientated subjects.
</item 2>

<item 3>
National Senior Certificate
Citizenship and Responsibility , this section teaches learners about their rights and their duties as citizens . Assessment ( edit ) There are three types of
subjects : General Subjects An end - of - year examination : 75 % of the total mark Portfolio ( Continuous Assessment ) : 25 % of the total mark Practical
Subjects An end - of - year examination : 50 % of the total mark Portfolio ( Continuous Assessment ) : 25 % of the total mark Practical component : 25 % of the
total mark Language Subjects An end - of - year examination : 50 % of the total mark Portfolio ( Continuous Assessment ) : 25 % of the total mark Oral
Examination : 25 % of the total mark Continuous Assessment ( CASS ) includes all the tests , examinations , tasks , activities , orals and projects done
throughout the year . Matric results are usually out of 400 marks .
</item 3>

<item 4>
National Senior Certificate
( Written reply to parliamentary question in 2011 ) `` The compensation applies to learners whose first language is neither English nor Afrikaans and who offer
an African language as their first language . They receive an additional 5 % on their non-language subjects . The measure was first introduced in 1999 by the
South African Certification Council . '' ( Written reply to parliamentary question in 2007 , ) Dr Sizwe Mabizela , Chairperson of Council , Umalusi has provided
further explanation : `` This is the most misunderstood concept in this country . In terms of the policy on language compensation , learners who offer an
African language as Home Language and do not offer Afrikaans or English as Home language qualify for a 5 % language compensation on the mark they have obtained
in a non-language subject .
</item 4>

<item 5>
National Senior Certificate
: 10 - 2007 ) Jump up ^ Politicsweb , 3 January 2011 , `` Umalusi explains matric mark adjustments '' , Dr Sizwe Mabizela , 03 January 2011 Jump up ^
http://nbt.ac.za/ About NBT Retrieved 21 March 2011 Jump up ^ http://wced.pgwc.gov.za/circulars/circulars08/e33_08.html External links ( edit ) Department of
Education official site IEB official site Retrieved from `` https://en.wikipedia.org/w/index.php?title=National_Senior_Certificate&oldid=837110357 '' Categories
: Education in South Africa Secondary school qualifications Hidden categories : Articles needing additional references from April 2010 All articles needing
additional references Talk Contents About Wikipedia Add links This page was last edited on 18 April 2018 , at 19 : 39 . About Wikipedia
</item 5>

<item 6>
National Senior Certificate
National Senior certificate - wikipedia National Senior certificate Jump to : navigation , search This article needs additional citations for verification .
Please help improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( April 2010 ) ( Learn how and
when to remove this template message ) The National Senior Certificate or NSC is a high school diploma and is the main school - leaving certificate in South
Africa . This certificate is commonly known as the matriculation ( matric ) certificate , as grade 12 is the matriculation grade . The NSC , previously known as
the Further Education and Training Certificate or FETC , replaced the Senior Certificate with effect from 2008 and was phased in starting with grade 10 in 2006
. Pupils study at least seven subjects , including two compulsory official South African languages , either Mathematics or Mathematical Literacy , Life
Orientation and three elective subjects . Subjects are all taken on the same level - there is no higher or standard grade as in the past . There are three pass
levels , each with different pass requirements.
</item 6>

<item 7>
National Senior Certificate
For example a learner who obtains a mark of zero ( 0 ) out of 300 will obtain 5 % of zero ( which is zero ) for language compensation ; a learner who obtains 10
out of 300 will receive 5 % of 10 , which is 0 , 5 marks , for language compensation ; a learner who obtains 100 out of 300 marks will obtain an additional 5
marks for language compensation . '' This kind of compensation has some impact on pass rates , but does provide significant impact at the upper end of the scale
, affecting those applying for admission to university . For example , a qualifying learner obtaining 95 % would receive 95 x 1.05 = 99.75 % ( which rounds to
100 % ) . A learner obtaining 40 % would receive 40 x 1.05 = 42.0 % . University entrance ( edit ) In addition to minimum grades required in each subject ,
universities either set their own entrance tests and / or use the National Benchmark Tests ( NBT ) .
</item 7>

<item 8>
National Senior Certificate
To study for a bachelor 's degree at a South African university requires that the applicant has at least an NSC endorsed by Umalusi , with a pass of 30 % in the
chosen university 's language of learning and teaching , as well as a level 4 or higher in the following list of designated , 19 - credit subjects : Accounting
Agricultural Sciences Business Studies Dramatic Arts Economics Engineering Graphics and Design Geography History Consumer Studies Information Technology
Languages ( one language of learning and teaching at a higher education institution and two other recognised language subjects ) Life Sciences Mathematics
Mathematical Literacy Music Physical Sciences Religion Studies Visual Arts See also ( edit ) High school # South Africa Matriculation in South Africa
Independent Examinations Board References ( edit ) Jump up ^ SOUTH AFRICAN NATIONAL ASSEMBLY : QUESTION 757 , DATE OF PUBLICATION OF INTERNAL QUESTION PAPER :
11 / 03 / 2011 ( INTERNAL QUESTION PAPER : 06 / 2011 ) Jump up ^ SOUTH AFRICAN NATIONAL ASSEMBLY : WRITTEN REPLY TO QUESTION 469 , DATE OF PUBLICATION IN
INTERNAL QUESTION PAPER : 26 - 03 - 2007 ( INTERNAL QUESTION PAPER NO.
</item 8>

<item 9>
National Senior Certificate
Matric results are usually out of 400 marks . Level system ( 1 to 7 ) ( edit ) Level 7 : 80 - 100 % ( Outstanding achievement ) Level 6 : 70 - 79 % (
Meritorious achievement ) Level 5 : 60 - 69 % ( Substantial achievement ) Level 4 : 50 - 59 % ( Moderate achievement ) Level 3 : 40 - 49 % ( Adequate
achievement ) Level 2 : 30 - 39 % ( Elementary achievement ) Level 1 : 0 - 29 % ( Not achieved - Fail ) The 5 % language compensation ( edit ) Language
compensation is described by several sources : `` To the final mark is added the language compensation , which is 5 % of the mark attained by the candidate for
all non-language subjects , for candidates whose mother tongue is not English or Afrikaans . The 5 % compensates learners for the disadvantage suffered by these
candidates being instructed in a language that is not their mother tongue . ''
</item 9>

<item 10>
National Senior Certificate
There are three pass levels , each with different pass requirements. First is a pass with Higher certificate , one must obtain 40 % in home language , 40 % in
two other subjects and must obtain 30 % in three other subjects . A learner who passed matric with higher certificate level , can not enroll for a University
Degree nor a Diploma at an institute of higher learning . The mean mark in any subject is usually about 55 . Only a small proportion of candidates score an ' A
' in any subject ( from as little as 2 % to a maximum of about 10 % in subjects taken by highly select groups . ) A further 8 -- 15 % are likely to gain a ' B '
and about 20 -- 25 % achieve a ' C ' grade . The National Senior Certificate is a group certificate and records an aggregate mark . The Department of Basic
Education has responsibility for general educational policy to be implemented by nine provincial education departments and private providers such as the
Independent Examinations Board ( IEB ) . There are nine provincial examination boards and three independent boards , of which the IEB is the biggest . The IEB
operates on a national level catering primarily for independent schools .
</item 10>
=== END ITEM id=e5ce0c53c4391514 ===

=== ITEM id=54eeec1c51e07967 ===
Gold sentences and facts:
Sentence 0: On Earth , Ivan 's Ecto - Morphicons are completely unearthed , and he unleashes them on Angel Grove , ordering the parents to commit suicide at the construction site . Fred , Bulk , Skull and the other kids head to the construction site to save their parents . The Rangers return with their new animal - themed Ninja Zords and destroy one of Ivan 's Ecto - Morphicons . Ivan takes control of the other and battles the Rangers himself . The Rangers lead Ivan into space right into the path of Ryan 's Comet , which destroys him . His destruction breaks the hypnosis and the parents are reunited with their children . The Rangers then use the Great Power to restore the Command Center and resurrect Zordon .
  Fact 0.0: On Earth, Ivan's Ecto-Morphicons are completely unearthed.
  Fact 0.1: Ivan unleashes the Ecto-Morphicons on Angel Grove.
  Fact 0.2: Ivan orders the parents to commit suicide at the construction site.
  Fact 0.3: Fred, Bulk, Skull and the other kids head to the construction site to save their parents.
  Fact 0.4: The Rangers return with their new animal-themed Ninja Zords.
  Fact 0.5: The Rangers destroy one of Ivan's Ecto-Morphicons.
  Fact 0.6: Ivan takes control of the other Ecto-Morphicon.
  Fact 0.7: Ivan battles the Rangers himself.
  Fact 0.8: The Rangers lead Ivan into space, right into the path of Ryan's Comet.
  Fact 0.9: Ryan's Comet destroys Ivan.
  Fact 0.10: Ivan's destruction breaks the hypnosis.
  Fact 0.11: The parents are reunited with their children.
  Fact 0.12: The Rangers use the Great Power to restore the Command Center.
  Fact 0.13: The Rangers use the Great Power to resurrect Zordon.

Retrieved items:
<item 1>
Mighty Morphin Power Rangers: The Movie
( 2005 ) Episodes Mystic Force ( 2006 ) Episodes Operation Overdrive ( 2007 ) Episodes Jungle Fury ( 2008 ) Episodes RPM ( 2009 ) Episodes Saban Brands Samurai
& Super Samurai ( 2011 -- 2012 ) Season 1 Megaforce & Super Megaforce ( 2013 -- 2014 ) Season 1 Dino Charge & Dino Super Charge ( 2015 -- 2016 ) Season 1 Ninja
Steel & Super Ninja Steel ( 2017 -- 2018 ) Season 1 Beast Morphers ( 2019 ) Films Mighty Morphin Power Rangers : The Movie ( 1995 ) Turbo : A Power Rangers
Movie ( 1997 ) Saban 's Power Rangers ( 2017 ) Web series Power Rangers Hyperforce ( 2017 -- 2018 ) Video games Mighty Morphin Power Rangers ( 1994 ) The Movie
( 1995 ) The Fighting Edition ( 1995 ) Zeo : Battle Racers ( 1996 ) Lightspeed Rescue ( 2000 ) Time Force ( 2001 ) Wild Force ( 2002 ) Ninja Storm ( 2003 ) Dino
Thunder ( 2004
</item 1>

<item 2>
Mighty Morphin Power Rangers: The Movie
For the 2017 film , see Power Rangers ( film ) . Mighty Morphin Power Rangers : The Movie Theatrical release poster Directed by Bryan Spicer Produced by Haim
Saban Shuki Levy Suzanne Todd Screenplay by Arne Olsen Story by John Kamps Arne Olsen Starring Karan Ashley Johnny Yong Bosch Steve Cardenas Jason David Frank
Amy Jo Johnson David Yost Jason Narvy Paul Schrier Paul Freeman Music by Graeme Revell Cinematography Paul Murphy Edited by Wayne Wahrman Production company
Saban Entertainment Toei Company Distributed by 20th Century Fox Release date June 30 , 1995 ( 1995 - 06 - 30 ) Running time 95 minutes Country United States
Language English Budget $15 million Box office $66.4 million Mighty Morphin Power Rangers : The Movie ( also known as Power Rangers : The Movie ) is a 1995
American superhero film based on the television series Mighty Morphin Power Rangers . It stars the ensemble cast of Karan Ashley , Johnny Yong Bosch , Steve
Cardenas , Jason David Frank , Amy Jo Johnson , and David Yost alongside the villains cast from the original series and Paul Freeman as Ivan Ooze .
</item 2>

<item 3>
Mighty Morphin Power Rangers: The Movie
Mighty Morphin Power Rangers : the Movie - wikipedia Mighty Morphin Power Rangers : the Movie Jump to : navigation , search `` Power Rangers : The Movie ''
redirects here . For the 2015 fan film , see Power / Rangers . For the 2017 film , see Power Rangers ( film ) .
</item 3>

<item 4>
Mighty Morphin Power Rangers: The Movie
He felt that the characters , with the exception of Ivan Ooze , lacked personalities , and that the scenes of monsters rampaging through the city hearkened back
to the worst Japanese monster films . Mick LaSalle of the San Francisco Chronicle found the fights `` only adequately choreographed , '' called the battle in
the climax `` a complete disaster '' and stating that it made no sense in timing , that protagonists were not very intelligent , and the actors playing them
unremarkable . Home media ( edit ) This section does not cite any sources . Please help improve this section by adding citations to reliable sources . Unsourced
material may be challenged and removed . ( May 2015 ) ( Learn how and when to remove this template message ) The film was released on VHS and LaserDisc in late
1995 and then as a double feature with 1997 's Turbo : A Power Rangers Movie on a double - sided DVD in 2001 by 20th Century Fox Home Entertainment . Bonus
features included a theatrical trailer and a `` Making Of '' featurette . The film was then released separately on a single - sided DVD in 2003 .
</item 4>

<item 5>
Mighty Morphin Power Rangers: The Movie
The New York Times . Retrieved 2010 - 08 - 07 . Jump up ^ `` Mighty Morphin Power Rangers : The Movie '' . RogerEbert.com . June 30 , 1995 . Retrieved 2010 - 08
- 07 . Jump up ^ Lasalle , Mick ( 1995 - 06 - 30 ) . `` Mighty Mindless ' Rangers ' '' . San Francisco Chronicle . Retrieved 2010 - 08 - 25 . External links (
edit ) Mighty Morphin Power Rangers : The Movie on IMDb Power Rangers Media TV series Saban Entertainment Mighty Morphin Power Rangers ( 1993 -- 1995 ; 2010 )
Season 1 2010 re-version Alien Rangers ( 1996 ) Zeo ( 1996 ) Episodes Turbo ( 1997 ) Episodes in Space ( 1998 ) Episodes Lost Galaxy ( 1999 ) Episodes
Lightspeed Rescue ( 2000 ) Episodes Time Force ( 2001 ) Episodes Disney Wild Force ( 2002 ) Episodes Ninja Storm ( 2003 ) Episodes Dino Thunder ( 2004 )
Episodes S.P.D. (
</item 5>

<item 6>
Mighty Morphin Power Rangers: The Movie
Much like the television season that followed the release , it used concepts from the Japanese Super Sentai series Kyoryu Sentai Zyuranger , Ninja Sentai
Kakuranger . It is the first Power Rangers production from Saban Entertainment not to feature any archived footage from Super Sentai . Filming took place in and
around Sydney and Queensland , Australia . It was released by 20th Century Fox on June 30 , 1995 . The film grossed $66.43 million worldwide , but received
middling and polarizing reviews from critics . Contents ( hide ) 1 Plot 2 Cast 2.1 Voices 3 Music 4 Adaptations in other media 5 Release 5.1 Critical reception
6 Home media 7 References 8 External links Plot ( edit ) The Power Rangers participate with Bulk and Skull in a charity skydive for the Angel Grove observatory
, in anticipation of Ryan 's Comet which is scheduled to pass by in two days . Bulk and Skull miss the target landing zone and accidentally land on a
construction site where a giant egg has been unearthed .
</item 6>

<item 7>
Mighty Morphin Power Rangers: The Movie
Fred , Bulk , Skull and the other kids head to the construction site to save their parents . The Rangers return with their new animal - themed Ninja Zords and
destroy one of Ivan 's Ecto - Morphicons . Ivan takes control of the other and battles the Rangers himself . The Rangers lead Ivan into space right into the
path of Ryan 's Comet , which destroys him . His destruction breaks the hypnosis and the parents are reunited with their children . The Rangers then use the
Great Power to restore the Command Center and resurrect Zordon . In a mid-credits scene , Goldar briefly lounges in Zedd 's throne being served by Mordant only
to panic when Zedd and Rita appear having been released after Ivan was destroyed . Cast ( edit ) Karan Ashley as Aisha Campbell , The Yellow Ranger Johnny Yong
Bosch as Adam Park , The Black Ranger Steve Cardenas as Rocky DeSantos , The Red Ranger Jason David Frank as Tommy Oliver , The White Ranger Amy Jo Johnson as
Kimberly Hart , The Pink Ranger David Yost as Billy Cranston ,
</item 7>

<item 8>
Mighty Morphin Power Rangers: The Movie
Lord Zedd , Rita Repulsa , Goldar , and Mordant arrive at the construction site and crack open the egg , releasing Ivan Ooze after 6,000 years , a morphological
being who ruled Earth with an iron fist before he was overthrown by Zordon and a group of young warriors . Ivan lays siege to the Rangers ' Command Center and
incapacitates Zordon , robbing the Rangers of their powers . As the Rangers return to the Command Center , they find it destroyed and Zordon dying . Zordon 's
assistant Alpha 5 sends the Rangers to the distant planet Phaedos to obtain the Great Power and save Zordon . On the Moon , Ivan usurps Rita and Zedd , trapping
them in a snow globe , then sends his Tengu warriors to Phaedos and begins building an army . He uses children to bring his ooze to their parents , and it
hypnotizes them into becoming his workforce to dig up his Ecto - Morphicon Titans , twin war machines built during his reign .
</item 8>

<item 9>
Mighty Morphin Power Rangers: The Movie
( 2003 ) Dino Thunder ( 2004 ) S.P.D. ( 2005 ) Super Legends ( 2007 ) Legacy Wars ( 2017 ) Others `` Day of the Dumpster '' ( series premiere ) `` Go Go Power
Rangers '' ( theme song ) `` Power Rangers '' ( song ) A Rock Adventure World Tour Live on Stage Collectible Card Game Comics Mighty Morphin Justice League /
Power Rangers Home video Mighty Morphin Universe Rangers Planets Races Villains Zords Related articles Cast members Power / Rangers Super Sentai Power Rangers (
South Korea ) Power Rangers Dino Force Brave Corporate Haim Saban Saban Entertainment Saban Brands Shuki Levy Tony Oliver Disney Bandai Namco Entertainment
Bandai Hasbro Related Saban series VR Troopers Masked Rider Beetleborgs ( Big Bad & Metallix ) Ninja Turtles : The Next Mutation Mystic Knights of Tir Na Nog
Related DIC series Superhuman Samurai Syber - Squad Tattooed Teenage Alien Fighters from Beverly Hills Related Adness series Kamen Rider : Dragon Knight Films
directed by Bryan Spicer Mighty Morphin Power Rangers : The Movie ( 1995 ) McHale 's Navy ( 1997 ) For Richer or Poorer ( 1997 ) Retrieved from `` https://en.
</item 9>

<item 10>
Mighty Morphin Power Rangers: The Movie
When Fred Kelman , a friend of the Rangers ' , discovers his father missing , he finds him working at the construction site and discovers Ivan 's plans . On
Phaedos , the Rangers are almost killed by the Tengu , but are rescued by Dulcea , Phaedos ' Master Warrior . She initially tells them to leave for their own
safety , but after hearing of Zordon 's plight , she agrees to help them and takes them to an ancient ruined temple where the Rangers will have to overcome
obstacles to acquire the power of the Ninjetti . Dulcea awakens each Rangers ' animal spirit : Aisha is the bear , Rocky is the ape , Billy is the wolf ,
Kimberly is the crane , Adam is the frog and Tommy is the falcon . The Rangers make their way to the Monolith housing the Great Power , defeating its guardians
, and retrieve the Great Power . On Earth , Ivan 's Ecto - Morphicons are completely unearthed , and he unleashes them on Angel Grove , ordering the parents to
commit suicide at the construction site . Fred , Bulk , Skull and the other kids head to the construction site to save their parents .
</item 10>
=== END ITEM id=54eeec1c51e07967 ===

=== ITEM id=097edb8350ba667c ===
Gold sentences and facts:
Sentence 0: R.H. Thomson as Matthew Cuthbert
  Fact 0.0: R.H. Thomson played the role of Matthew Cuthbert.

Retrieved items:
<item 1>
Anne (TV series)
Writing for Entertainment Weekly , Isabella Beidenharn expressed similar feelings , but , `` putting the source material aside , it 's a fine show on its own ''
, and she conceded that `` inventing a dark side might help Anne With an E fit into today 's TV landscape '' . Allison Keene , writing for Collider , agrees
that Anne is a good drama on its own terms , but allows it is `` only a fair adaptation '' of the novel , at its best in the home scenes : `` Anne with an E is
undeniably the most stylish adaptation we 've ever seen of Anne of Green Gables . But its desire to reveal more of Anne 's miserable past in order to be more
true to what the desperation of an orphan is like feels at odds with Montgomery 's story . ''
</item 1>

<item 2>
Anne (TV series)
IndieWire . Retrieved 17 June 2017 . Jump up ^ Chaney , Jen . `` Anne of Green Gables Fans , You Will Love Netflix 's Anne With an E '' . Vulture.com . New York
Magazine . Retrieved 17 June 2017 . Jump up ^ Genzlinger , Neil ( May 11 , 2017 ) . `` Review : ' Anne With an E ' Is a Rewarding Return to Green Gables '' .
The New York Times . Retrieved 17 June 2017 . Jump up ^ Hirschlag , Annie . `` ' Anne with an E ' Review : The Neflix Adpatation Hits the Nostalgia Button and
More '' . Mic.com . Mic ( media Company ) . Retrieved 17 June 2017 . Jump up ^ Nawaz , Saleema ( March 21 , 2017 ) . `` We asked an Anne of Green Gables
superfan to review the first episode of CBC 's new TV series '' . Toronto Life . Retrieved 4 June 2017 .
</item 2>

<item 3>
Anne (TV series)
When Matthew goes to pick the child up at the railway station , he finds not a boy , but a high - spirited and talkative girl , Anne Shirley . At first , the
Cuthberts are inclined to send her back , particularly after Marilla 's brooch goes missing , and Anne , in despair , runs away . Reconciled and settled in her
new home and school , Anne once again displays boundless enthusiasm which is nevertheless easily turned into despair when things go wrong , which they often do
. Slowly , her ebullient nature wins over those around her . Cast ( edit ) Main ( edit ) Amybeth McNulty as Anne Shirley Geraldine James as Marilla Cuthbert
R.H. Thomson as Matthew Cuthbert Dalila Bela as Diana Barry Lucas Jade Zumann as Gilbert Blythe Corrine Koslo as Rachel Lynde Aymeric Jett Montaz as Jerry
Baynard Others ( edit ) Jonathan Holmes as Mr. William Barry Helen Johns as Mrs. Eliza Barry Ryan Kiera Armstrong as Minnie Mary Barry Deborah Grover as
Josephine Barry Wayne Best as John Blythe Phillip Williams as Thomas Lynde David Ingram as Mr.
</item 3>

<item 4>
Anne (TV series)
Anne ( TV series ) - wikipedia Anne ( TV series ) Jump to : navigation , search Anne Also known as Anne with an E Genre Drama Created by Moira Walley - Beckett
Based on Anne of Green Gables by Lucy Maud Montgomery Screenplay by Moira Walley - Beckett Starring Amybeth McNulty Geraldine James R.H. Thomson Lucas Jade
Zumann Dalila Bela Corrine Koslo Aymeric Jett Montaz Opening theme `` Ahead by a Century '' by The Tragically Hip Country of origin Canada Original language ( s
) English No. of seasons No.
</item 4>

<item 5>
Anne (TV series)
`` CBC and Netflix renew reimagined Anne of Green Gables '' . Toronto Star . Retrieved August 3 , 2017 . Jump up ^ `` Anne ( with an E ) : Season Two Filming
Begins for CBC and Netflix Series '' . TV Series Finale . November 24 , 2017 . ^ Jump up to : `` Netflix 's ' Anne of Green Gables ' Adaptation Finds Its Anne
Shirley , Casts 2 Other Roles '' . Deadline . October 21 , 2016 . ^ Jump up to : `` Netflix 's ' Anne Of Green Gables ' Adaptation Adds Three To Cast '' .
Deadline.com . Retrieved November 21 , 2016 . Jump up ^ `` Maureen McCormick To Guest In ' The Guest Book ' ; Lucas Jade Zumann Joins ' Anne ' '' . Deadline.com
. Retrieved November 18 , 2016 . Jump up ^ Hunt , Nigel ( March 19 , 2017 ) . `` ' Ai n't your grandmother 's Anne ' : new series gives gritty Green Gables amid
glut of Anne adaptations '' . CBC News .
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
Jump up ^ `` Anne with an E : Season 1 '' . Metacritic . Retrieved 17 June 2017 . Jump up ^ Ashby , Emily . `` Anne with an E '' . Common Sense Media .
Retrieved 17 June 2017 . Jump up ^ Cerny , Tasha . `` ANNE WITH AN E Review : `` Episodes 1 - 3 '' `` . www.thetrackingboard.com . Retrieved 18 June 2017 . Jump
up ^ Inhat , Gwen . `` Anne With An E offers a winning , darker take on a familiar tale '' . A.V. Club . Retrieved 18 June 2017 . Jump up ^ Schneller , Johanna
( March 20 , 2017 ) . `` This Anne of Green Gables hints at miseries : What I 'm Watching '' . Toronto Star . access - date = requires url = ( help ) Jump up ^
Nguyen , Hanh . `` ' Anne with an E ' Review : A ' Breaking Bad ' Producer Updates ' Anne of Green Gables ' for Our Darker Times '' .
</item 7>

<item 8>
Anne (TV series)
7 `` Wherever You Are Is My Home '' Amanda Tapping Moira Walley - Beckett April 30 , 2017 ( 2017 - 04 - 30 ) N / A The Cuthberts vow to do whatever it takes to
save the farm which reminds Anne of the strength of friendship and love . Reception ( edit ) Critical response ( edit ) The series has achieved a rating of 88 %
on Rotten Tomatoes , the site 's critical consensus stating : `` Anne with an E uses its complex central character to offer a boldly stylish , emotionally
resonant spin on classic source material that satisfies in its own right . '' The series has received a rating of 79 on Metacritic based on fifteen reviews ,
indicating `` generally favorable reviews '' . Emily Ashby , writing for Common Sense Media , calls the series an `` exceptional '' and `` spectacular ''
interpretation , giving it four out of five stars . Tasha Cerny , contributor for the Tracking Board , praises the cinematography as lush and colourful , the
characters vibrant , and the plot `` surprisingly thrilling for a story about a young girl living in a small secluded community in the late nineteenth century .
</item 8>

<item 9>
Anne (TV series)
Numeris . Jump up ^ Brioux , Bill ( @ BillBriouxTV ) ( April 10 , 2017 ) . `` SUN onites SNET NHL TOR - COL 1181k VAN -- EDM 766k TSN CURL 1026k GLO NCISLA 1Mk
CBC ANNE 654k STORY US 546k CITY SHOTS 623k CTV SAVE HOPE 548k '' ( Tweet ) -- via Twitter . Jump up ^ Brioux , Bill ( @ BillBriouxTV ) ( April 24 , 2017 ) . ``
SJN o'nites SNet NHL TOR - WASH Game 6 3467k OTT - BOS 1284k OMNI2 TOR - WASH 121k CTV SAVE HOPE 686K CBC ANNE 656K STORY OF US 450K '' ( Tweet ) -- via Twitter
. Jump up ^ `` Anne with an E ( 2017 - ) '' . Rotten Tomatoes . Retrieved 4 January 2018 . Jump up ^ `` Anne with an E : Season 1 '' . Metacritic .
</item 9>

<item 10>
Anne (TV series)
'' She went on to call Anne Shirley an `` accidental feminist '' , and how she `` really wanted to tell this story now . '' Broadcast and Release ( edit ) The
series initially premiered on March 19 , 2017 , on CBC and aired on a weekly basis , the season finale airing on April 30 , 2017 . The series debuted on Netflix
on May 12 , 2017 , under the title Anne with an E . Episodes ( edit ) Season 1 ( 2017 ) ( edit ) No . Title Directed by Written by Original air date Can .
viewers ( millions ) `` Your Will Shall Decide Your Destiny '' Niki Caro Moira Walley - Beckett March 19 , 2017 ( 2017 - 03 - 19 ) 0.999 When a miscommunication
brings a girl , Anne Shirley , to Green Gables instead of a boy , Marilla and Matthew Cuthbert are faced with a life - changing decision .
</item 10>
=== END ITEM id=097edb8350ba667c ===

=== ITEM id=5b6159b6ed864562 ===
Gold sentences and facts:
Sentence 0: Natural Born Killers Theatrical release poster Directed by Oliver Stone Produced by Jane Hamsher Don Murphy Clayton Townsend Screenplay by Richard Rutowski Oliver Stone David Veloz Story by Quentin Tarantino Starring Woody Harrelson Juliette Lewis Robert Downey , Jr . Tommy Lee Jones Tom Sizemore Cinematography Robert Richardson Edited by Hank Corwin Brian Berdan Production company Regency Enterprises Distributed by Warner Bros . Release date August 26 , 1994 ( 1994 - 08 - 26 ) Running time 119 minutes Country United States Language English Japanese Navajo Budget $34 million Box office $50.3 million
  Fact 0.0: Natural Born Killers has a theatrical release poster.
  Fact 0.1: Natural Born Killers was directed by Oliver Stone.
  Fact 0.2: Natural Born Killers was produced by Jane Hamsher.
  Fact 0.3: Natural Born Killers was produced by Don Murphy.
  Fact 0.4: Natural Born Killers was produced by Clayton Townsend.
  Fact 0.5: Natural Born Killers has a screenplay by Richard Rutowski.
  Fact 0.6: Natural Born Killers has a screenplay by Oliver Stone.
  Fact 0.7: Natural Born Killers has a screenplay by David Veloz.
  Fact 0.8: Natural Born Killers has a story by Quentin Tarantino.
  Fact 0.9: Natural Born Killers stars Woody Harrelson.
  Fact 0.10: Natural Born Killers stars Juliette Lewis.
  Fact 0.11: Natural Born Killers stars Robert Downey, Jr.
  Fact 0.12: Natural Born Killers stars Tommy Lee Jones.
  Fact 0.13: Natural Born Killers stars Tom Sizemore.
  Fact 0.14: Natural Born Killers has cinematography by Robert Richardson.
  Fact 0.15: Natural Born Killers was edited by Hank Corwin.
  Fact 0.16: Natural Born Killers was edited by Brian Berdan.
  Fact 0.17: Natural Born Killers was produced by Regency Enterprises.
  Fact 0.18: Natural Born Killers was distributed by Warner Bros.
  Fact 0.19: Natural Born Killers has a release date of August 26, 1994.
  Fact 0.20: Natural Born Killers has a running time of 119 minutes.
  Fact 0.21: Natural Born Killers is from the United States.
  Fact 0.22: Natural Born Killers has English, Japanese and Navajo languages.
  Fact 0.23: Natural Born Killers had a budget of $34 million.
  Fact 0.24: Natural Born Killers had a box office of $50.3 million.
Sentence 1: Natural Born Killers is a 1994 American satirical black comedy crime film directed by Oliver Stone and starring Woody Harrelson , Juliette Lewis , Robert Downey Jr. , Tom Sizemore , and Tommy Lee Jones . The film was released in the United States on August 26 , 1994 . The film tells the story of two victims of traumatic childhoods who became lovers and mass murderers , and are irresponsibly glorified by the mass media .
  Fact 1.0: Natural Born Killers is a 1994 American satirical black comedy crime film.
  Fact 1.1: Natural Born Killers was directed by Oliver Stone.
  Fact 1.2: Natural Born Killers stars Woody Harrelson.
  Fact 1.3: Natural Born Killers stars Juliette Lewis.
  Fact 1.4: Natural Born Killers stars Robert Downey Jr.
  Fact 1.5: Natural Born Killers stars Tom Sizemore.
  Fact 1.6: Natural Born Killers stars Tommy Lee Jones.
  Fact 1.7: Natural Born Killers was released in the United States on August 26, 1994.
  Fact 1.8: The film tells the story of two victims of traumatic childhoods.
  Fact 1.9: The two victims became lovers.
  Fact 1.10: The two victims became mass murderers.
  Fact 1.11: The two victims are irresponsibly glorified by the mass media.

Retrieved items:
<item 1>
Natural Born Killers
Emil Reingold Pruitt Taylor Vince as Deputy Warden Kavanaugh Joe Grifasi as Deputy Sheriff Duncan Homolka Everett Quinton as Deputy Wurlitzer Marshall Bell as
Deputy Peter Crombie as Intense cop Grand L. Bush as Prison inmate Louis Lombardi as Deputy `` Sparky '' Dale Dye as Dale Wrigley Corey Everson as Mallory Knox
in TV reconstruction O - Lan Jones as Mabel James Gammon ( uncredited ) as Redneck buddy Mark Harmon ( uncredited ) as Mickey Knox in TV reconstruction Adrien
Brody and David Pasquesi ( uncredited ) as Gale 's cameramen Arliss Howard ( uncredited ) as Owen Traft , Mickey and Mallory 's Guardian angel / The Demon
Deleted scenes Ashley Judd as Grace Mulberry Rachel Ticotin as Prosecutor Wanda Bisbing Peter and David Paul as Simon and Norman Hun Denis Leary and Bret Hart
as Prison inmates Production Natural Born Killers was based upon a screenplay written by Quentin Tarantino , in which a married couple suddenly decide to go on
a killing spree .
</item 1>

<item 2>
Natural Born Killers
( 2017 ) Retrieved from `` https://en.wikipedia.org/w/index.php?title=Natural_Born_Killers&oldid=819366442 '' Categories : 1994 films 1990s crime films 1990s
drama films American films American black comedy films American crime drama films American independent films American satirical films English - language films
Film duos Films about psychopaths Films directed by Oliver Stone Films set in the 1990s Films set in Arizona Films set in Illinois Films set in Indiana Films
set in New Mexico Films shot in multiple formats Films with live action and animation Incest in film Matricide in fiction Obscenity controversies in film
Patricide in fiction Regency Enterprises films Road movies Screenplays by Oliver Stone Screenplays by Quentin Tarantino Serial killer films Venice Grand Special
Jury Prize winners Warner Bros.
</item 2>

<item 3>
Natural Born Killers
Style Natural Born Killers is shot and edited in a frenzied and psychedelic style consisting of black and white , animation , and other unusual color schemes ,
and employing a wide range of camera angles , filters , film stocks , lenses , and special effects . Much of the film is told via parodies of television shows ,
including a scene ( I Love Mallory ) presented in the style of a sitcom about a dysfunctional family . Commercials which were commonly on the air at the time of
the film 's release make brief , intermittent appearances . In his DVD director 's commentary , Stone goes into great detail about the look of the film ,
explaining scene by scene why a particular look was chosen for a particular scene . According to Hollywood.com , Natural Born Killers is a satirical crime film
, while Foster Hirsch deemed it `` the crime film as would - be social and cultural satire '' . Stone considered it his road film , specifically naming Bonnie
and Clyde as a source of inspiration .
</item 3>

<item 4>
Natural Born Killers
The film was released in the United States on August 26 , 1994 . The film tells the story of two victims of traumatic childhoods who became lovers and mass
murderers , and are irresponsibly glorified by the mass media . The film is based on an original screenplay by Quentin Tarantino that was heavily revised by
writer David Veloz , associate producer Richard Rutowski , and director Stone ; Tarantino received story credit . Jane Hamsher , Don Murphy , and Clayton
Townsend produced the film , with Arnon Milchan , Thom Mount , and Stone as executive producers . The film was released theatrically on August 26 , 1994 in the
United States , February 24 , 1995 and premiered on Venice Film Festival on August 29 , 1994 . The film was a box office success , grossing over $50 million
against a production budget of $34 million , and received generally mixed to positive reviews from critics , who praised the cast 's performances , the movie 's
humor , plot , and combination of action and romance , but found the film much too violent and graphic .
</item 4>

<item 5>
Natural Born Killers
Nytimes.com ( 1996 - 04 - 14 ) . Retrieved on 2014 - 05 - 22 . Jump up ^ Natural Born Killers ( Unrated Director 's Cut ) ( Blu - ray ) : Woody Harrelson ,
Juliette Lewis , Tom Sizemore , Oliver Stone : Movies & TV . Amazon.com . Retrieved on 2014 - 05 - 22 . Jump up ^ `` ' Natural Born Killers ' Is Banned in
Ireland '' . NY Times . Retrieved 2013 - 05 - 26 . Jump up ^ `` Irish Film Institute - Natural Born Killers '' . Irish Film Institute . Retrieved 2013 - 05 - 26
. Jump up ^ ( 1 ) Archived March 19 , 2015 , at the Wayback Machine . Jump up ^ `` Natural Born Killers given video release '' . The Guardian . London . May 16
, 2001 .
</item 5>

<item 6>
Natural Born Killers
Natural Born Killers - Wikipedia Natural Born Killers Jump to : navigation , search Not to be confused with Natural Born Killaz . Natural Born Killers
Theatrical release poster Directed by Oliver Stone Produced by Jane Hamsher Don Murphy Clayton Townsend Screenplay by Richard Rutowski Oliver Stone David Veloz
Story by Quentin Tarantino Starring Woody Harrelson Juliette Lewis Robert Downey , Jr . Tommy Lee Jones Tom Sizemore Cinematography Robert Richardson Edited by
Hank Corwin Brian Berdan Production company Regency Enterprises Distributed by Warner Bros . Release date August 26 , 1994 ( 1994 - 08 - 26 ) Running time 119
minutes Country United States Language English Japanese Navajo Budget $34 million Box office $50.3 million Natural Born Killers is a 1994 American satirical
black comedy crime film directed by Oliver Stone and starring Woody Harrelson , Juliette Lewis , Robert Downey Jr. , Tom Sizemore , and Tommy Lee Jones . The
film was released in the United States on August 26 , 1994 .
</item 6>

<item 7>
Natural Born Killers
McClusky and his guards are then quickly massacred by hordes of inmates . Mickey and Mallory steal a van and kill the last guard ; Owen 's fate is unknown .
Escaping to a rural location , they give a final interview to Gale , whose ear they have seemingly removed ( implied , but as a camera track reveals ) , before
they tell him he must die also . He attempts various arguments to change their minds , finally appealing to their trademark practice of leaving one survivor ;
Mickey informs him they are leaving a witness to tell the tale , his camera . Gale accepts his fate and extends his arms as if on a cross as they shoot him dead
while his unattended camera continues to roll . The couple is shown several years later , in an RV , with Mickey driving and a pregnant Mallory watching their
two children play . Cast Main article : List of Natural Born Killers characters Woody Harrelson as Mickey Knox Juliette Lewis as Mallory Wilson Knox Robert
Downey , Jr. as Wayne Gale Tom Sizemore as Detective Jack Scagnetti Tommy Lee Jones as Warden Dwight McClusky Rodney Dangerfield as Ed Wilson Edie McClurg as
Mrs.
</item 7>

<item 8>
Natural Born Killers
Dre `` Born Bad '' -- Juliette Lewis `` Fall of the Rebel Angels '' ( Edit ) -- Sergio Cervetti `` Forkboy '' -- Lard `` Batonga in Batongaville '' ( Edit ) --
`` A Warm Place '' ( Edit ) -- Nine Inch Nails `` Allah , Mohammad , Char , Yaar '' -- Nusrat Fateh Ali Khan `` The Future '' ( Edit ) -- Leonard Cohen `` What
Would U Do ? '' -- Tha Dogg Pound Tracks 10 , 13 , 18 , 20 , 23 , and 25 are assembled from various recordings and dialogue from the film . See also Film in the
United States portal 1990s portal Charles Starkweather References Jump up ^ `` NATURAL BORN KILLERS ( 18 ) '' . British Board of Film Classification . December
20 , 1994 . Retrieved September 25 , 2014 . ^ Jump up to : `` Natural Born Killers ( 1994 ) '' . Box Office Mojo. 1994 - 10 - 18 . Retrieved 2016 - 01 - 06 .
</item 8>

<item 9>
Natural Born Killers
The Guardian . London . May 16 , 2001 . Jump up ^ 25 Most Controversial Movies Ever , Entertainment Weekly ( archived at Internet Archive ) , 2010 - 01 - 06 ,
Archived from the original on January 6 , 2010 , retrieved 2016 - 06 - 10 CS1 maint : Unfit url ( link ) Further reading Hamsher , Jane ( 1998 ) . Killer
Instinct . Broadway . Hanley , Jason . ( 2001 ) `` Natural Born Killers : Music and Image in Postmodern Film , '' in Postmodern Music / Postmodern Thought ,
Routledge . ed . Joseph Auner and Judy Lochhead , pp. 335 -- 359 . External links Wikiquote has quotations related to : Natural Born Killers Natural Born
Killers on IMDb Natural Born Killers at Box Office Mojo Natural Born Killers at Rotten Tomatoes Natural Born Killers at Metacritic `` Natural Born Killers :
Beyond Good and Evil '' , by Heidi Nelson Hochenedel , Ph. D .
</item 9>

<item 10>
Natural Born Killers
As Stone himself says , `` In its own way , Natural Born Killers is ultimately a very optimistic film about the future . It 's about freedom , and the ability
of every human being to get it . '' Release Box Office This section needs expansion . You can help by adding to it . ( November 2014 ) In its opening weekend ,
Natural Born Killers grossed a total of $11,166,687 in 1,510 theaters . As of January 12 , 2007 , the film had grossed a total of $50,282,766 domestically ,
compared to its $34 million budget . Critical response The film received a mixed critical response . As of December 1 , 2017 , the review aggregator Rotten
Tomatoes records an average response of 46 % , based on 37 reviews . However , Metacritic records a score of 74 out of 100 based on 20 reviews . Roger Ebert of
the Chicago Sun - Times gave the film four stars out of four and wrote , `` Seeing this movie once is not enough . The first time is for the visceral experience
, the second time is for the meaning . ''
</item 10>
=== END ITEM id=5b6159b6ed864562 ===

=== ITEM id=6aa69af5799949bf ===
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
17 ( K. 129 ) No. 18 ( K. 130 ) No. 19 ( K. 132 ) No. 20 ( K. 133 ) No. 21 ( K. 134 ) No. 22 ( K. 162 ) No. 23 ( K. 181 ) No. 24 ( K. 182 ) No. 25 ( K. 183 )
No. 26 ( K. 184 ) No. 27 ( K. 199 ) No. 28 ( K. 200 ) No. 29 ( K. 201 ) No. 30 ( K. 202 ) No. 31 ( K. 297 , Paris ) No. 32 ( K. 318 ) No. 33 ( K. 319 ) No. 34 (
K. 338 ) No. 35 ( K. 385 , Haffner ) No. 36 ( K. 425 , Linz ) No. 37 ( K. 444 ) No. 38 ( K. 504 , Prague ) No. 39 ( K. 543 ) No.
</item 1>

<item 2>
Symphony No. 40 (Mozart)
Passage quoted may be viewed on line at Google Books : ( 5 ) . Schoenberg , Arnold ( 1954 , rvs. 1969 ) . Structural Functions of Harmony . New York : W.W.
Norton & Company . Analyzes the wide - ranging development sections of both outer movements at some length . Swafford , Jan ( 1997 ) . Johannes Brahms : A
Biography . New York : Alfred A. Knopf . Zaslaw , Neal ( 1983 ) . Introductory notes to a recording of the 31st and 40th Symphonies made by Christopher Hogwood
and the Academy of Ancient Music , Oiseau - Lyre 410 - 197 - 2 . Zaslaw , Neal ( 1994 ) . `` Mozart as Working Stiff '' . In James M. Morris , ed . On Mozart ,
. Washington : Woodrow Wilson Center Press . Zaslaw , Neal , and William Cowdery ( 1990 ) .
</item 2>

<item 3>
Symphony No. 40 (Mozart)
Franz Schubert likewise copied down the music of Mozart 's minuet , and the minuet of his Fifth Symphony strongly evokes Mozart 's . Zaslaw has suggested that a
passage late in Joseph Haydn 's oratorio The Seasons ( 1801 ) , a meditation on death , quotes the second movement of the 40th Symphony and was included by
Haydn as a memorial to his long - dead friend . First recording ( edit ) The first known recording of the 40th Symphony is by the Victor Recording Company and
issued in 1915 under the title `` Symphony in G Minor '' . The Victor Concert Orchestra performed under the direction of conductor Walter B. Rogers . Media (
edit ) The following files contain a digital recording of a performance of the 40th Symphony by the Fulda Symphonic Orchestra . The performance took place on
March 18 , 2001 in the Orangerie in Fulda , Germany . 1st movement Molto allegro 2nd movement Andante 3rd movement Menuetto , Allegretto - Trio 4th movement
Allegro assai Problems playing these files ? See media help .
</item 3>

<item 4>
Symphony No. 40 (Mozart)
Zaslaw , Neal , and William Cowdery ( 1990 ) . The Compleat Mozart : A Guide to the Musical Works of Wolfgang Amadeus Mozart . New York : W.W. Norton & Company
. Extracts cited are viewable online at Google Books : ( 6 ) . External links ( edit ) Wikimedia Commons has media related to Symphony No. 40 ( Mozart ) .
Versions of the score : Sinfonie in g minor K. 550 ( 1st version ) : Score and critical report ( in German ) in the Neue Mozart - Ausgabe Sinfonie in g minor K.
550 ( 2nd version ) : Score in the Neue Mozart - Ausgabe Symphony No. 40 : Scores at the International Music Score Library Project ( IMSLP ) Kalmus miniature
editions , from the William and Gayle Cook Music Library at the Indiana University Jacobs School of Music A Public Domain version by the Columbia University
Orchestra at TEOC Classical Music Radio : Symphony No.
</item 4>

<item 5>
Symphony No. 40 (Mozart)
Reception ( edit ) This work has elicited varying interpretations from critics . Robert Schumann regarded it as possessing `` Grecian lightness and grace '' .
Donald Francis Tovey saw in it the character of opera buffa . Almost certainly , however , the most common perception today is that the symphony is tragic in
tone and intensely emotional ; for example , Charles Rosen ( in The Classical Style ) has called the symphony `` a work of passion , violence , and grief '' .
Although interpretations differ , the symphony is unquestionably one of Mozart 's most greatly admired works , and it is frequently performed and recorded .
Influence ( edit ) Ludwig van Beethoven knew the symphony well , copying out 29 bars from the score in one of his sketchbooks . As Gustav Nottebohm observed in
1887 , the copied bars appear amid the sketches for Beethoven 's Fifth Symphony , whose third movement begins with a pitch sequence similar to that of Mozart 's
finale ( see example above ) .
</item 5>

<item 6>
Symphony No. 40 (Mozart)
Nikolaus Harnoncourt conjectured that Mozart composed the three symphonies as a unified work , pointing , among other things , to the fact that the Symphony No.
40 , as the middle work , has no introduction ( unlike No. 39 ) and does not have a finale of the scale of No. 41 's . The 40th symphony exists in two versions
, differing primarily in that one includes parts for a pair of clarinets ( with suitable adjustments made in the other wind parts ) . Most likely , the clarinet
parts were added in a revised version . The autograph scores of both versions were acquired in the 1860s by Johannes Brahms , who later donated the manuscripts
to the Gesellschaft der Musikfreunde in Vienna , where they reside today . Premiere ( edit ) As Neal Zaslaw has pointed out , writers on Mozart have often
suggested -- or even asserted -- that Mozart never heard his 40th Symphony performed . Some commentators go further , suggesting that Mozart wrote the symphony
( and its companions , Nos.
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
The contrapuntal opening bars of this movement appear thus in keyboard reduction : The minuet begins with an angry , cross-accented hemiola rhythm and a pair of
three - bar phrases , as shown in the following piano reduction : Various commentators have asserted that while the music is labelled `` minuet '' , it would
hardly be suitable for dancing . The contrasting gentle trio section , in G major , alternates the playing of the string section with that of the winds . The
fourth movement opens with a series of rapidly ascending notes outlining the tonic triad illustrating what is commonly referred to as the Mannheim rocket . The
movement is written largely in eight - bar phrases , following the general tendency toward rhythmic squareness in the finales of classical - era symphonies . A
remarkable modulating passage in which every tone in the chromatic scale but one is played , strongly destabilising the key , occurs at the beginning of the
development section . The single note left out is in fact a G ( the tonic ) . The symphony typically has a duration of about 25 minutes . Reception ( edit )
This work has elicited varying interpretations from critics .
</item 8>

<item 9>
Symphony No. 40 (Mozart)
40 The apartment where Mozart wrote his last three Symphonies : Michael Lorenz , `` Mozart 's Apartment on the Alsergrund '' Article online Wolfgang Amadeus
Mozart Biography Biographies Birthplace Grand tour Name Nationality Scatology Smallpox Italy Berlin Prague Death Music Köchel catalogue List of compositions
Concert arias , songs , canons Dances Horn concertos Masses Operas Piano concertos Works for solo piano Sonatas Symphonies Compositional method Family Leopold
Mozart Anna Maria Mozart Maria Anna Mozart ( Nannerl ) Constanze Mozart Maria Anna Thekla Mozart ( Bäsle ) Franz Xaver Wolfgang Mozart Karl Thomas Mozart Johann
Georg Mozart Joseph Lange Cäcilia Weber Josepha Weber Aloysia Weber Sophie Weber Georg Nissen Family trees Influences Beethoven Catholic Church Freemasonry
Haydn Salieri Related Mozart in popular culture Book ( hide ) Symphonies by Wolfgang Amadeus Mozart Unnumbered K. 16a ( Odense ) K. 19a K. 45a ( Old Lambach )
K. 76 K. 45b K. 74g K. 81 K. 95 K. 97 K.
</item 9>

<item 10>
Symphony No. 40 (Mozart)
Symphony No. 40 ( Mozart ) - wikipedia Symphony No. 40 ( Mozart ) Jump to : navigation , search Symphony No. 40 in G minor , KV. 550 was written by Wolfgang
Amadeus Mozart in 1788 . It is sometimes referred to as the `` Great G minor symphony '' , to distinguish it from the `` Little G minor symphony '' , No. 25 .
The two are the only extant minor key symphonies Mozart wrote . Contents ( hide ) 1 Composition 2 Premiere 3 The music 4 Reception 5 Influence 6 First recording
7 Media 8 Notes 9 References 10 External links Composition ( edit ) The date of completion of this symphony is known exactly , since Mozart in his mature years
kept a full catalog of his completed works ; he entered the 40th Symphony into it on 25 July 1788 . Work on the symphony occupied an exceptionally productive
period of just a few weeks during which time he also completed the 39th and 41st symphonies ( 26 June and 10 August , respectively ) .
</item 10>
=== END ITEM id=6aa69af5799949bf ===

=== ITEM id=95c85bc8231d1850 ===
Gold sentences and facts:
Sentence 0: Rose Stagg ( Valene Kane ) , a woman who had a sexual relationship with Paul whilst at university . During a night of passion , Paul tried to strangle her to death . As the Belfast Strangler comes to the attention of the public , Stagg notices the similarities and contacts her close friend Reed Smith . Her husband was unaware the attack had taken place . She is later kidnapped and held captive by Spector . She is later found unconscious in a car trunk . During series three , Stagg reveals that her sexual relationship with Paul was entirely consensual , and that , during sex , he strangled her so aggressively she had to be revived .
  Fact 0.0: Rose Stagg (played by Valene Kane) had a sexual relationship with Paul while at university.
  Fact 0.1: During a night of passion, Paul tried to strangle Rose Stagg to death.
  Fact 0.2: As the Belfast Strangler comes to the attention of the public, Rose Stagg notices the similarities.
  Fact 0.3: Rose Stagg contacts her close friend Reed Smith.
  Fact 0.4: Rose Stagg's husband was unaware the attack had taken place.
  Fact 0.5: Rose Stagg is later kidnapped and held captive by Spector.
  Fact 0.6: Rose Stagg is later found unconscious in a car trunk.
  Fact 0.7: During series three, Rose Stagg reveals that her sexual relationship with Paul was entirely consensual.
  Fact 0.8: During sex, Paul strangled Rose Stagg so aggressively that she had to be revived.
Sentence 1: Actor Character Series 1 Series 2 Series 3 Total Appearances 5 6 7 8 9 10 11 12 13 14 15 16 17 Gillian Anderson DSI Stella Gibson 17 Jamie Dornan Paul Spector 17 John Lynch ACC Jim Burns 17 Aisling Franciosi Katie Benedetto N / A 16 Bronagh Waugh Sally Ann Spector N / A 15 Niamh McGrady PC Danielle Ferrington N / A N / A 15 Stuart Graham DCI Matthew Eastwood N / A N / A N / A 13 Bronágh Taggart DC Gail McNally N / A 12 Valene Kane Rose Stagg N / A N / A N / A N / A 11 Emmett J. Scanlan DC Glen Martin N / A N / A N / A 10 Archie Panjabi Prof. Reed Smith N / A N / A 9 Richard Clements DC Rick Turner N / A N / A N / A N / A 9 Colin Morgan DS Tom Anderson N / A 9 Karen Hassan Annie Brawley N / A N / A N / A N / A 8 Jonjo O'Neill Tom Stagg N / A N / A N / A 7 Nick Lee Ned Callan N / A N / A N / A 6 Brian Milligan James Tyler N / A N / A N / A 6 Seainin Brennan Liz Tyler N / A N / A N / A N / A 5 Ben Peel DS James Olson N / A N / A 5 Ian McElhinney Morgan Monroe N / A N / A 5 Aidan McArdle Sean Healy N / A 5 Ruth Bradley Wallace N / A 5 Michael McElhatton DI Rob Breedlove N / A Frank McCusker DCI Garrett Brink N / A N / A Aisling Bea Kiera Sheridan N / A N / A Barry Ward Dr. Patrick Spencer N / A N / A Krister Henriksson Dr. August Larson N / A Laura Donnelly Sarah Kay N / A N / A Richard Coyle Dr. Joe O'Donnell N / A N / A N / A Denise Gough Dr. Alison Walden N / A N / A Seán McGinley Fr . Peter Jensen N / A N / A Genevieve O'Reilly DCI Joan Kinkead N / A N / A Martin McCann David Alvarez N / A N / A
  Fact 1.0: The cast table lists each actor's character and appearances across series one to three, with episode numbers 5 to 17 as column headers.
  Fact 1.1: Gillian Anderson plays DSI Stella Gibson, with 17 total appearances.
  Fact 1.2: Jamie Dornan plays Paul Spector, with 17 total appearances.
  Fact 1.3: John Lynch plays ACC Jim Burns, with 17 total appearances.
  Fact 1.4: Aisling Franciosi plays Katie Benedetto, with 16 total appearances.
  Fact 1.5: Bronagh Waugh plays Sally Ann Spector, with 15 total appearances.
  Fact 1.6: Niamh McGrady plays PC Danielle Ferrington, with 15 total appearances.
  Fact 1.7: Stuart Graham plays DCI Matthew Eastwood, with 13 total appearances.
  Fact 1.8: Bronágh Taggart plays DC Gail McNally, with 12 total appearances.
  Fact 1.9: Valene Kane plays Rose Stagg, with 11 total appearances.
  Fact 1.10: Emmett J. Scanlan plays DC Glen Martin, with 10 total appearances.
  Fact 1.11: Archie Panjabi plays Prof. Reed Smith, with 9 total appearances.
  Fact 1.12: Richard Clements plays DC Rick Turner, with 9 total appearances.
  Fact 1.13: Colin Morgan plays DS Tom Anderson, with 9 total appearances.
  Fact 1.14: Karen Hassan plays Annie Brawley, with 8 total appearances.
  Fact 1.15: Jonjo O'Neill plays Tom Stagg, with 7 total appearances.
  Fact 1.16: Nick Lee plays Ned Callan, with 6 total appearances.
  Fact 1.17: Brian Milligan plays James Tyler, with 6 total appearances.
  Fact 1.18: Seainin Brennan plays Liz Tyler, with 5 total appearances.
  Fact 1.19: Ben Peel plays DS James Olson, with 5 total appearances.
  Fact 1.20: Ian McElhinney plays Morgan Monroe, with 5 total appearances.
  Fact 1.21: Aidan McArdle plays Sean Healy, with 5 total appearances.
  Fact 1.22: Ruth Bradley plays Wallace, with 5 total appearances.
  Fact 1.23: Michael McElhatton plays DI Rob Breedlove.
  Fact 1.24: Frank McCusker plays DCI Garrett Brink.
  Fact 1.25: Aisling Bea plays Kiera Sheridan.
  Fact 1.26: Barry Ward plays Dr. Patrick Spencer.
  Fact 1.27: Krister Henriksson plays Dr. August Larson.
  Fact 1.28: Laura Donnelly plays Sarah Kay.
  Fact 1.29: Richard Coyle plays Dr. Joe O'Donnell.
  Fact 1.30: Denise Gough plays Dr. Alison Walden.
  Fact 1.31: Seán McGinley plays Fr. Peter Jensen.
  Fact 1.32: Genevieve O'Reilly plays DCI Joan Kinkead.
  Fact 1.33: Martin McCann plays David Alvarez.

Retrieved items:
<item 1>
List of The Fall characters
`` BBC Two Sets Premiere Date For ' The Fall ' '' . TVWise . Retrieved 1 May 2013 . External links ( edit ) The Fall at BBC Programmes The Fall on RTÉ The Fall
on Netflix The Fall on IMDb The Fall at Rotten Tomatoes The Fall Trailer at ZDF Enterprises The Fall on Facebook Retrieved from ``
https://en.wikipedia.org/w/index.php?title=List_of_The_Fall_characters&oldid=811099612 '' Categories : Lists of drama television characters Lists of British
television series characters Lists of Irish television series characters Hidden categories : Use dmy dates from January 2017 Use British English from January
2017 BBC programme ID not in Wikidata Talk Contents About Wikipedia Add links This page was last edited on 19 November 2017 , at 14 : 36 . About Wikipedia
</item 1>

<item 2>
When Calls the Heart
He later tells her that his son died , as well as his wife . However , his wife , Nora , comes to confront him in season 2 's `` Heart 's Desire . '' Henry
Gowen is a cold , cynical man who formerly owned the coal mine . After the mine was shut down , Henry became the mayor of Hope Valley and still creates a cold ,
cynical presence whenever people are around him . Henry is highly disliked and feared in the town . He frequently uses bribery to get what he wants , which is
never for the good of the town . Rosemary `` Rosie '' LeVeaux Coulter is a flamboyant , enthusiastic actress from New York . She was briefly engaged to Jack
Thornton , but they both realized that it would n't work out . Rosie immediately disliked Elizabeth and viewed her as `` the competition , '' but eventually
came to really care about Elizabeth , even if she has a hard time showing it . At the beginning of season 2 , Rosemary began working at the cafe with Abigail
and the two quickly became friends . Spending time with Abigail helped tone down some of Rosemary 's enthusiasm and made her a more likable person .
</item 2>

<item 3>
List of The Fall characters
Gibson relies on her heavily during the investigative process . Rose Stagg ( edit ) Rose Stagg ( Valene Kane ) , a woman who had a sexual relationship with Paul
whilst at university . During a night of passion , Paul tried to strangle her to death . As the Belfast Strangler comes to the attention of the public , Stagg
notices the similarities and contacts her close friend Reed Smith . Her husband was unaware the attack had taken place . She is later kidnapped and held captive
by Spector . She is later found unconscious in a car trunk . During series three , Stagg reveals that her sexual relationship with Paul was entirely consensual
, and that , during sex , he strangled her so aggressively she had to be revived . DC Glen Martin ( edit ) DC Glen Martin ( Emmett Scanlan ) , a capable
detective with a somewhat boyish attitude towards his work . Martin is an invaluable member of Gibson 's team and , during series two , takes on a more
prominent role in the decision making process . He is partnered with DC Gail McNally , and is fiercely protective of her , though not to a fault .
</item 3>

<item 4>
When Calls the Heart
Rosemary even came to like Elizabeth , even if only on an acquaintance level . Truthfully , she cares about everyone in Hope Valley , especially Lee , with whom
she shares an mutual attraction . When Rosie is not spending time with Lee , talking to Abigail , or dreaming about the glamorous life of an actress , she can
often be found somewhere writing her weekly advice column for Hope Valley 's newspaper ! In season 3 , Rosemary 's friendship with Elizabeth deepened . Instead
of viewing her as an adversary , she finds a friend in Elizabeth . Throughout season 3 , Rosemary 's relationship with Lee becomes more romantic . In `` Heart
Of A Hero , '' seeing that Lee needs to spend more time at the sawmill to help the new settlers find jobs , Rosemary tells Lee that the theatre does n't have to
be built . He insists , but she says how important his job is . He smiles and tells her he will one day build her theatre , telling her that she is `` the best
part of his life . '' The couple shares in their first kiss . In `` A Gentle Heart , '' Lee is injured .
</item 4>

<item 5>
When Calls the Heart
She has a daughter Rosaleen . In season 1 's `` A Telling Silence , '' Rosaleen has trouble in school because she misses her father so much . When she runs away
, Elizabeth finds her in the mine , feeling guilty that she did n't bring her father his lunch the day of the disaster because she wanted to play with a friend
instead . She fears her mother will blame her , but Elizabeth convinces her that Molly will `` say that she loves her more than life itself . '' Molly is
reunited with her daughter , to the joy of the town 's people . In season 2 's `` With All My Heart , '' Lee feels it 's time to `` play hard to get '' with
Rosemary , so he enlists help from Molly . She pretends to be interested in him romantically . This ploy works and Rosemary admits that she does in fact like
him . Molly is a kind woman and was among the first to welcome Elizabeth to town . Clara Stanton is the wife of Abigail 's late son Peter , and is a sweet ,
quiet , and sensitive young woman .
</item 5>

<item 6>
List of The Fall characters
List of the Fall characters - wikipedia List of the Fall characters Jump to : navigation , search The Fall , starring Gillian Anderson , is a British - Irish
drama series produced by Artists Studio and created by Allan Cubitt . It premiered in the Republic of Ireland at 21 : 30 on 12 May 2013 , and in the UK on BBC
Two at 21 : 00 on 13 May 2013 . The series features Jamie Dornan , with John Lynch , Bronagh Waugh , Stuart Graham , and Niamh McGrady . Contents ( hide ) 1
Cast overview 2 Main characters 2.1 Det . Supt . Stella Gibson 2.2 Peter Paul Spector 2.3 Asst . Ch . Constable Jim Burns 2.4 Katie Benedetto 2.5 Sally Ann
Spector 2.6 PC Danielle Ferrington 2.7 Det . Ch . Insp . Matthew Eastwood 2.8 DC Gail McNally 2.9 Rose Stagg 2.10 DC Glen Martin 2.11 Professor Reed Smith 2.12
DC Rick Turner 2.13 Det .
</item 6>

<item 7>
List of The Fall characters
Asst . Ch . Constable Jim Burns ( edit ) ACC Jim Burns ( John Lynch ) , the Assistant Chief Constable of the PSNI in charge of operations , and the Gold
Commander of Musicman . Burns had an affair with Stella several years previously , and , despite being married , still appears to be infatuated with her . He 's
a skilled officer and knows how to balance politics and police - work . He is initially hesitant about linking the murders , but later provides the
investigation with all the resources it requires . He is , at first , believed to be linked to the PSNI corruption scandal , though this is later disproved .
During series 2 , his lust for Stella becomes more apparent when he attempts to force himself on her whilst drunk . This drinking continues into series three ,
and he resigns his commission during the series finale . Katie Benedetto ( edit ) Katerina `` Katie '' Benedetto ( Aisling Franciosi ) , a 15 - year - old
schoolgirl . She babysits the Spectors ' children and - over the course of doing so - develops an affinity for Paul Spector .
</item 7>

<item 8>
Republic of Doyle
Jake realizes he has other problems too ; his mischievous older brother , Christian makes an unexpected visit to his house , greeting him in the kitchen . After
Christian offends Jake , their relationship swiftly intensifies . The last episode of the season shows the characters ' struggle to keep their republic strong
as their relationships fall apart . Season two ( edit ) Season Two begins with Rose , Mal 's partner , dealing with the consequences of his recent heart attack
. Meanwhile , Jake and Leslie take initiatives to move past the hiccups in their relationship and move on from one another . Tinny , Jake 's niece , embarks on
an investment in order to raise personal funds , meanwhile , Mal discovers a secret that Rose has kept from him ; her mother stole her engagement ring . Jake
learns what Leslie sees their relationship as . As the season progresses , so does Tinny 's income business along with the amount of friends she makes . Jake as
well as the viewers are introduced to Allison , the crown attorney . Meanwhile , Mal and Rose experience issues in their marriage but are quickly able to
recover and move forward .
</item 8>

<item 9>
List of The Fall characters
After leaving Paul for a short time , believing he slept with Katie , the two reconcile and she flees Ireland with him . In series 2 , they return . Sally Ann
is pregnant with her third child , though she suffers a miscarriage . Following her arrest for assisting an offender , Sally attempts to kill her children . Pc
Danielle Ferrington ( edit ) PC Danielle Ferrington ( Niamh McGrady ) , a Police Constable first seen attending a breaking and entering call at Sarah Kay 's
home . She blames herself for Kay 's murder and tells Gibson this , resulting in her being recruited by Stella to work with Operation Musicman . Dani is a
skilled investigator , and proves herself invaluable to the team . She speaks openly of her homosexuality to bisexual Gibson , and as a result the two develop
both a professional and a personal bond . She later returns to patrol , believing she can do the most good on the street . It is in this capacity that she is
involved in a gunfight with Spector and , following her suspension from front - line duty , is re-recruited by Stella . Det . Ch . Insp .
</item 9>

<item 10>
List of The Fall characters
Ch . Insp . Joan Kinkead 2.33 David Alvarez 3 References 4 External links Cast overview ( edit ) Actor Character Series 1 Series 2 Series 3 Total Appearances 5
6 7 8 9 10 11 12 13 14 15 16 17 Gillian Anderson DSI Stella Gibson 17 Jamie Dornan Paul Spector 17 John Lynch ACC Jim Burns 17 Aisling Franciosi Katie Benedetto
N / A 16 Bronagh Waugh Sally Ann Spector N / A 15 Niamh McGrady PC Danielle Ferrington N / A N / A 15 Stuart Graham DCI Matthew Eastwood N / A N / A N / A 13
Bronágh Taggart DC Gail McNally N / A 12 Valene Kane Rose Stagg N / A N / A N / A N / A 11 Emmett J. Scanlan DC Glen Martin N / A N / A N / A 10 Archie Panjabi
Prof.
</item 10>
=== END ITEM id=95c85bc8231d1850 ===
