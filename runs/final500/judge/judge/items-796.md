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

=== ITEM id=e181fcfc2077fa8e ===
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
During the day , four townspeople were killed , including the mayor . Realizing his escape was cut , Brown took nine of his captives and moved into the smaller
engine house , which would come to be known as John Brown 's Fort . The raiders blocked entry of the windows and doors and traded sporadic gunfire with
surrounding forces . At one point Brown sent out his son , Watson , and Aaron Dwight Stevens with a white flag , but Watson was mortally wounded and Stevens was
shot and captured . The raid was rapidly failing . One of Brown 's men , William H. Leeman , panicked and made an attempt to flee by swimming across the Potomac
River , but he was shot and fatally injured while doing so . During the intermittent shooting , Brown 's other son , Oliver , was also hit ; he died after a
brief period . At around 3 : 00 p.m. a militia company led by Captain E.G. Alburtis arrived by train from Martinsburg , Virginia . Most of the militia members
were Baltimore & Ohio Railroad employees . The militia forced the raiders inside the engine house .
</item 1>

<item 2>
John Brown's raid on Harpers Ferry
2007 On Sunday night , October 16 , 1859 , Brown left four of his men behind as a rear - guard : his son , Owen Brown , Barclay Coppock , and Frank Meriam ; he
led the rest into the town of Harpers Ferry , Virginia . Brown detached a party under John Cook Jr. to capture Colonel Lewis Washington , great - grandnephew of
George Washington , at his nearby Beall - Air estate , some of his slaves , and two relics of George Washington : a sword allegedly presented to Washington by
Frederick the Great and two pistols given by Marquis de Lafayette , which Brown considered talismans . The party carried out its mission and returned via the
Allstadt House , where they took more hostages . Brown 's main party captured several watchmen and townspeople in Harpers Ferry . Brown 's men needed to capture
the weapons and escape before word could be sent to Washington . The raid was going well for Brown 's men . They cut the telegraph wire and seized a Baltimore &
Ohio train passing through . A free black man was the first casualty of the raid .
</item 2>

<item 3>
John Brown's raid on Harpers Ferry
Contents ( hide ) 1 Brown 's preparation 2 Advance knowledge of raid 3 The raid 3.1 October 16 3.2 October 17 3.3 October 18 3.4 October 19 4 Aftermath 5
Casualties 5.1 John Brown 's raiders 5.2 Others 6 Liberated slaves 7 See also 8 Notes 8.1 Citations 8.2 Bibliography 9 External links Brown 's preparation (
edit ) John Brown rented the Kennedy Farmhouse , with a small cabin nearby , 4 miles ( 6.4 km ) north of Harpers Ferry near the community of Dargan in
Washington County , Maryland , and took up residence under the name Isaac Smith . Brown came with a small group of men minimally trained for military action .
His group included 18 men besides himself ( 13 white men , 5 black men ) . Northern abolitionist groups sent 198 breech - loading . 52 caliber Sharps carbines (
`` Beecher 's Bibles '' ) and 950 pikes ( obtained in late September from Charles Blair of Collinsville Axe Co.
</item 3>

<item 4>
John Brown's raid on Harpers Ferry
Greene later recounted what events occurred next : Quicker than thought I brought my saber down with all my strength upon ( Brown 's ) head . He was moving as
the blow fell , and I suppose I did not strike him where I intended , for he received a deep saber cut in the back of the neck . He fell senseless on his side ,
then rolled over on his back . He had in his hand a short Sharpe 's cavalry carbine . I think he had just fired as I reached Colonel Washington , for the Marine
who followed me into the aperture made by the ladder received a bullet in the abdomen , from which he died in a few minutes . The shot might have been fired by
someone else in the insurgent party , but I think it was from Brown . Instinctively as Brown fell I gave him a saber thrust in the left breast . The sword I
carried was a light uniform weapon , and , either not having a point or striking something hard in Brown 's accouterments , did not penetrate . The blade bent
double . In three minutes , all of the raiders still alive were taken prisoner and the action was over .
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
, Collinsville , CT ) , in preparation for the raid . The United States Armory was a huge complex of buildings that manufactured small arms for the U.S. Army (
1801 -- 1861 ) , with an Arsenal ( storehouse ) that was thought to contain 100,000 muskets and rifles at the time . An illustration of Brown trying to persuade
abolitionist Frederick Douglass to join him in the raid on Harpers Ferry . Douglass refused . Brown attempted to attract more black recruits . He tried
recruiting Frederick Douglass as a liaison officer to the slaves in a meeting held in a quarry at Chambersburg , Pa . It was at this meeting that ex-slave ``
Emperor '' Shields Green consented to join with John Brown on his attack on the United States Armory , Green stating to Douglass `` I believe I will go with the
old man '' . Douglass declined , indicating to Brown that he believed the raid was a suicide mission . The plan was `` an attack on the federal government ''
that `` would array the whole country against us ... You will never get out alive '' , he warned .
</item 6>

<item 7>
John Brown's raid on Harpers Ferry
Regardless , one was killed trying to escape across the Potomac River , the other was wounded and later died in the Charles Town jail . ) 9 other civilians were
wounded . Marines Private Luke Quinn ( Killed during the storming of the engine house . ) Buried in Harpers Ferry Catholic Cemetery on Rte. 340 . Private
Matthew Ruppert ( Shot in the face during the storming of the engine house ; survived . ) Liberated slaves ( edit ) Mayor Beckham 's Will Book called for the
liberation of Isaac Gilbert , his wife , and three children upon his death . When Edwin Coppock killed Beckham the five slaves were free . See also ( edit )
List of incidents of civil unrest in the United States Notes ( edit ) Citations ( edit ) ^ Jump up to : `` John Brown 's raid on Harpers Ferry '' . History.com
. Retrieved 1 May 2015 . Jump up ^ In many books the town is called `` Harper 's Ferry . '' For example , `` Col.
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
John Brown 's raid on Harpers Ferry - wikipedia John Brown 's raid on Harpers Ferry Jump to : navigation , search John Brown 's Raid on Harper 's Ferry Part of
pre-Civil War conflicts Harper 's Weekly illustration of U.S. Marines attacking John Brown 's `` Fort '' Date October 16 -- 18 , 1859 Location Harpers Ferry , (
West ) Virginia Result U.S. victory Belligerents United States Abolitionist Insurgents Commanders and leaders Robert E. Lee Israel Greene Jeb Stuart John Brown
Strength 88 U.S. Marines Unknown number of Virginia Militia and Maryland Militia 8 white men 12 free black men 1 freed slave 1 fugitive slave Casualties and
losses U.S.
</item 10>
=== END ITEM id=e181fcfc2077fa8e ===

=== ITEM id=50f87e9bfb7f0186 ===
Gold sentences and facts:
Sentence 0: The village of New Earswick was built as a genuine mixed community . There was housing for both workers and managers , in a green setting with gardens for each home with its own 2 fruit trees . It was founded by the York philanthropist , Joseph Rowntree , who was quoted as saying , `` I do not want to establish communities bearing the stamp of charity but rather of rightly ordered and self governing communities '' . The first 28 houses were built between 1902 and 1904 by the architect Raymond Unwin , after which the Joseph Rowntree Village Trust ( now the Joseph Rowntree Housing Trust ) was established to continue building and manage the new village . The Joseph Rowntree Housing Trust is part of the Joseph Rowntree Foundation .
  Fact 0.0: New Earswick was built as a genuine mixed community.
  Fact 0.1: New Earswick had housing for both workers and managers.
  Fact 0.2: New Earswick is set in a green setting.
  Fact 0.3: Each home in New Earswick had a garden.
  Fact 0.4: Each home in New Earswick had its own 2 fruit trees.
  Fact 0.5: New Earswick was founded by the York philanthropist Joseph Rowntree.
  Fact 0.6: Joseph Rowntree said he did not want to establish communities bearing the stamp of charity but rather of rightly ordered and self governing communities.
  Fact 0.7: The first 28 houses of New Earswick were built between 1902 and 1904.
  Fact 0.8: The architect Raymond Unwin built the first 28 houses of New Earswick.
  Fact 0.9: The Joseph Rowntree Village Trust, now the Joseph Rowntree Housing Trust, was established after the first houses were built to continue building and manage the new village.
  Fact 0.10: The Joseph Rowntree Housing Trust is part of the Joseph Rowntree Foundation.
Sentence 1: As a result of the report , Joseph Rowntree 's conviction that it must be possible to provide better housing for people on low incomes led him to acquire 150 acres of land near the village of Earswick , two and a half miles to the north of the centre of York . The planner Raymond Unwin and the architect Barry Parker were commissioned to produce an overall plan for a new ' garden ' village and the detailed designs for its first houses . They also designed the garden cities of Letchworth and Welwyn Garden City .
  Fact 1.0: Joseph Rowntree's conviction followed from a report.
  Fact 1.1: Joseph Rowntree was convinced it must be possible to provide better housing for people on low incomes.
  Fact 1.2: Joseph Rowntree acquired 150 acres of land near the village of Earswick.
  Fact 1.3: The land near Earswick is two and a half miles north of the centre of York.
  Fact 1.4: The planner Raymond Unwin and the architect Barry Parker were commissioned to produce an overall plan for a new 'garden' village.
  Fact 1.5: Raymond Unwin and Barry Parker were commissioned to produce the detailed designs for the first houses of the new garden village.
  Fact 1.6: Raymond Unwin and Barry Parker designed the garden cities of Letchworth and Welwyn Garden City.

Retrieved items:
<item 1>
New Earswick
Jump up ^ `` New Earswick All Blacks '' . Archived from the original on 16 January 2008 . Jump up ^ Havard , Jane ( 1992 ) . A Bit On The Side . Menoshire . p.
1 . ISBN 0947981667 . External links ( edit ) `` New Earswick , York '' . The Joseph Rowntree Housing Trust . Retrieved 4 March 2017 .
</item 1>

<item 2>
New Earswick
There never has been , although the Sports Club obtained a drink 's license in the 1980s . Governance ( edit ) New Earswick is part of the Huntington & New
Earswick Ward of the City of York Council . As of 2010 it is represented by Keith Hyman , Keith Orrel and Carol Runciman who are all members of the local
Liberal Democrat Party . A Parish Council was created in 1934 consisting of ten elected councillors and a clerk . Community ( edit ) There is a local community
centre , the Folk Hall , which hosts activities such as keep - fit , yoga , snooker , and a junior youth club . New Earswick has recreational facilities ,
including a swimming pool ( due for closure without public consultation ) , tennis courts and football , rugby and cricket pitches at the New Earswick Sports
and Social Club . There were two open access play areas close to the Folk Hall , including a hard surface play area which have recently been bulldozed for a
development of houses .
</item 2>

<item 3>
New Earswick
The planner Raymond Unwin and the architect Barry Parker were commissioned to produce an overall plan for a new ' garden ' village and the detailed designs for
its first houses . They also designed the garden cities of Letchworth and Welwyn Garden City . The building of New Earswick created a balanced village community
where rents were kept low , but still represented a modest commercial return on the capital invested . Houses were open to any working people , not just
Rowntree employees . The village was to be a demonstration of good practice . The Trust Deed of the Joseph Rowntree Village Trust , which was set up in 1904 to
build and manage New Earswick , safeguarded generous open green space . All the grass verges were planted with trees after which almost all the roads are named
. The village was built with bricks that were made in the brickworks on the outskirts of New Earswick . From 1950 the brickyard , which closed down in the 1930s
, was developed into a nature reserve . Due to their religious beliefs , The Rowntree family decreed there would not be a public house in the village .
</item 3>

<item 4>
New Earswick
The first 28 houses were built between 1902 and 1904 by the architect Raymond Unwin , after which the Joseph Rowntree Village Trust ( now the Joseph Rowntree
Housing Trust ) was established to continue building and manage the new village . The Joseph Rowntree Housing Trust is part of the Joseph Rowntree Foundation .
The village had contemporaries such as Bournville , Saltaire , Port Sunlight and others . This was in sharp contrast to the slums that had developed in York and
other cities during the previous century , the deprivation of which had been revealed in Seebohm Rowntree 's report of 1901 , entitled Poverty : a study of town
life . It revealed appalling statistics of dark , overcrowded and insanitary housing . As a result of the report , Joseph Rowntree 's conviction that it must be
possible to provide better housing for people on low incomes led him to acquire 150 acres of land near the village of Earswick , two and a half miles to the
north of the centre of York .
</item 4>

<item 5>
New Earswick
The Joseph Rowntree Housing Trust . Retrieved 4 March 2017 . Retrieved from `` https://en.wikipedia.org/w/index.php?title=New_Earswick&oldid=826057861 ''
Categories : Model villages Villages and areas in the City of York Civil parishes in North Yorkshire Rowntree family Hidden categories : All articles with dead
external links Articles with dead external links from November 2017 Articles with permanently dead external links Use British English from October 2013 Use dmy
dates from October 2013 Articles with OS grid coordinates Coordinates on Wikidata All articles with unsourced statements Articles with unsourced statements from
April 2013 Talk Contents About Wikipedia Cebuano Gaeilge Nederlands Polski Svenska Edit links This page was last edited on 16 February 2018 , at 23 : 16 . Text
is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and
Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 5>

<item 6>
New Earswick
There is a library based in the Integrated Children 's Centre at the local primary school , a doctor 's surgery and a range of shops , including two general
stores , a pet shop and Post Office . As of 2017 the Post Office is situated within the Folk Hall . The village has 36 allotments in two areas , Willow Bank and
Sleeper Path . In the 1960s , the Folk Hall was a live music venue featuring the likes of Procol Harum and Pink Floyd . Geography ( edit ) The River Foss runs
along the eastern boundary of New Earswick with the York to Scarborough Line running to the west . The Nestle factory and grounds mark the southern border with
the northern border being the A1237 York Outer Ring Road . The village was built using bricks from the brickworks on the outskirts of the village . From 1950
the brickyard was developed into a nature reserve . The reserve is the home of New Earswick Angling Club , which was formed in 1948 .
</item 6>

<item 7>
New Earswick
New Earswick - wikipedia New Earswick Jump to : navigation , search New Earswick New Earswick New Earswick shown within North Yorkshire Population 2,737 ( 2011
) OS grid reference SE609555 Civil parish New Earswick Unitary authority City of York Ceremonial county North Yorkshire Region Yorkshire and the Humber Country
England Sovereign state United Kingdom Post town YORK Postcode district YO32 Dialling code 01904 Police North Yorkshire Fire North Yorkshire Ambulance Yorkshire
EU Parliament Yorkshire and the Humber UK Parliament York Outer List of places UK England Yorkshire 53 ° 59 ′ 31 '' N 1 ° 04 ′ 21 '' W ﻿ / ﻿ 53.992020 ° N
1.072380 ° W ﻿ / 53.992020 ; - 1.072380 Coordinates : 53 ° 59 ′ 31 '' N 1 ° 04 ′ 21 '' W ﻿ / ﻿ 53.992020 ° N 1.072380 ° W ﻿ / 53.992020 ; - 1.
</item 7>

<item 8>
New Earswick
Steve Webster , 10 times FIM Sidecar World Championships , world champion was born and raised in the village. . Gallery ( edit ) Views of New Earswick The Folk
Hall in New Earswick White Rose Avenue in New Earswick Westfield Beck behind housing on the west of New Earswick References ( edit ) Jump up ^ `` Civil Parish
population 2011 '' . Neighbourhood Statistics . Office for National Statistics . Retrieved 19 February 2016 . Jump up ^ `` Village History '' . Retrieved 12
December 2010 . Jump up ^ `` Village planning '' . Retrieved 12 December 2010 . Jump up ^ `` New Earswick : a village of new ideas '' . BBC . Retrieved 29 March
2014 . Jump up ^ `` Couoncillors '' . Retrieved 12 December 2010 . ^ Jump up to : `` Village Information '' ( PDF ) . Archived from the original ( PDF ) on 4
October 2013 . Retrieved 12 December 2010 .
</item 8>

<item 9>
New Earswick
Economy ( edit ) Unlike the villages of Port Sunlight , Saltaire and Bournville the village was not designed specifically for the workforce of the nearby Joseph
Rowntree Cocoa Works although some do earn their living at the factory which was taken over by Swiss firm , Nestle . Transport ( edit ) Transdev York operate a
bus service that stops in the village as part of the Askham Bar to Clifton Moor route . First Group operate two bus services that stop in the village as part of
the Wigginton to Acomb and Haxby to York City Centre routes . New Earswick was served by Earswick railway station on the York to Beverley Line between 1847 and
1965 . Education ( edit ) There are both primary and secondary schools in New Earswick . New Earswick Primary School was built in 1912 . The primary school is
one of eight Integrated Children 's Centres in York . Joseph Rowntree School is a secondary school and was built in 1941 . In 2006 , the Joseph Rowntree School
secured £ 27million for a complete rebuild that was completed in late 2009 .
</item 9>

<item 10>
New Earswick
072380 ° W ﻿ / 53.992020 ; - 1.072380 New Earswick is a model village and civil parish in the unitary authority of City of York in North Yorkshire , England ,
near the River Foss , north of York and south of Haxby . According to the 2001 census the parish had a population of 2,812 , reducing to 2,737 at the 2011
Census . Before 1996 it had been part of the Ryedale district . Contents ( hide ) 1 History 2 Governance 3 Community 4 Geography 5 Economy 6 Transport 7
Education 8 Religious sites 9 Sport 10 Gallery 11 References 12 External links History ( edit ) The village of New Earswick was built as a genuine mixed
community . There was housing for both workers and managers , in a green setting with gardens for each home with its own 2 fruit trees . It was founded by the
York philanthropist , Joseph Rowntree , who was quoted as saying , `` I do not want to establish communities bearing the stamp of charity but rather of rightly
ordered and self governing communities '' .
</item 10>
=== END ITEM id=50f87e9bfb7f0186 ===

=== ITEM id=d78bfeec764be522 ===
Gold sentences and facts:
Sentence 0: As a young woman , Rogue permanently absorbed the superhuman powers and the psyche of Carol Danvers , the original Ms. Marvel . This provided her with superhuman strength , stamina , durability , reflexes , speed and a 7th sense . She was able to repel bullets and fly at sub-sonic speeds , much like Ms. Marvel could . In addition , she possessed an amalgamated mutant human / Kree physiology that rendered her resistant to most toxins and poisons , with the added effect of making her virtually invulnerable .
  Fact 0.0: Rogue, as a young woman, permanently absorbed the superhuman powers of Carol Danvers.
  Fact 0.1: Rogue, as a young woman, permanently absorbed the psyche of Carol Danvers.
  Fact 0.2: Carol Danvers is the original Ms. Marvel.
  Fact 0.3: Rogue's absorption gave Rogue superhuman strength.
  Fact 0.4: Rogue's absorption gave Rogue superhuman stamina.
  Fact 0.5: Rogue's absorption gave Rogue superhuman durability.
  Fact 0.6: Rogue's absorption gave Rogue superhuman reflexes.
  Fact 0.7: Rogue's absorption gave Rogue superhuman speed.
  Fact 0.8: Rogue's absorption gave Rogue a 7th sense.
  Fact 0.9: Rogue was able to repel bullets.
  Fact 0.10: Rogue was able to fly at sub-sonic speeds.
  Fact 0.11: Rogue's ability to fly at sub-sonic speeds was much like Ms. Marvel's.
  Fact 0.12: Rogue possessed an amalgamated mutant human/Kree physiology.
  Fact 0.13: Rogue's amalgamated mutant human/Kree physiology made Rogue resistant to most toxins.
  Fact 0.14: Rogue's amalgamated mutant human/Kree physiology made Rogue resistant to most poisons.
  Fact 0.15: The added effect of Rogue's amalgamated physiology was that Rogue was virtually invulnerable.

Retrieved items:
<item 1>
Rogue (comics)
She battles the Avengers using her newly acquired powers . While at the Pentagon , she literally bumps into Carol Danvers , leading into a fight with Storm and
Wolverine of the X-Men . Though Rogue proves more than a match for them , the three of them escape the Pentagon . Later , with Mystique and Destiny , Rogue
attacks Angel and Dazzler , but is overpowered by Dazzler . Rogue develops a grudge against Dazzler for her controllable mutation and her relationship with
Angel of the X-Men . Rogue is defeated by Dazzler a few times before Dazzler is publicly revealed to be a mutant and goes into hiding . Sometime after this ,
Rogue , Mystique and Destiny encountered the half - human , half Dire Wraith entity called Hybrid , with whom the three formed an alliance with against the
Spaceknight , Rom . However , when Rogue seized an opportunity to use her power on him , she experienced both his loneliness and his nobility . This moment had
a profound effect on her and served as a crucial turning point in her life .
</item 1>

<item 2>
Rogue (comics)
She is also depicted as suffering side - effects from the mass terrigenesis , requiring regular injections to stop herself becoming ill or dying . When various
Avengers are brainwashed into civilian lives in the community of Pleasant Hill -- a S.H.I.E.L.D. project to ' reprogram ' supervillains into regular civilians
using fragments of a Cosmic Cube that has since gained sentience -- Rogue 's new identity of Claire sees through the deception thanks to telepathic training
provided by Professor X , allowing her mind to plant various clues that lead her to the rest of the team . Later on , the Red Skull attempts to mount a new
attack on the team , but Deadpool is able to resist the telepathic attack long enough to get Magneto 's old helmet on Rogue so that she can resist the Skull 's
telepathy . Rogue subsequently takes the Skull to a new facility where Beast is able to extract the uniquely Xavier elements from the Skull 's brain , depriving
him of Xavier 's telepathy , while Rogue and Johnny Storm incinerate the brain . Powers and abilities ( edit ) This section relies too much on references to
primary sources .
</item 2>

<item 3>
Rogue (comics)
Through meditation , Rogue learned how to suppress the random manifestations of most of these abilities , but found herself with a new ' default ' state for a
time , where she possessed Wolverine 's claws and healing factor , Cyclops ' optic blasts - requiring her to wear ruby quartz glasses - while also still
retaining her mutant and Ms. Marvel powers . Later , Sage used her own mutant abilities to ' jump - start ' Rogue 's power , giving her the control to manifest
any ability she had ever absorbed , whenever she wished to . It is generally accepted that after regaining her abilities after X-Treme X-Men , she has lost this
ability . Though Z'Cann was a telepath , Rogue never displayed this power ( though she did manifest telepathy , it was with the Phoenix energy form , so it was
most likely an echo of Jean Grey 's powers ) . Powerless ( edit ) For a time Rogue lost all of her powers , including her original mutant ones , after she was
fatally injured in a fight with the mysterious warrior Vargas . During this period , she displayed exceptional fighting skills and agility , though it was said
these were not superhuman in nature .
</item 3>

<item 4>
Rogue (comics)
Ms. Marvel powers ( edit ) As a young woman , Rogue permanently absorbed the superhuman powers and the psyche of Carol Danvers , the original Ms. Marvel . This
provided her with superhuman strength , stamina , durability , reflexes , speed and a 7th sense . She was able to repel bullets and fly at sub-sonic speeds ,
much like Ms. Marvel could . In addition , she possessed an amalgamated mutant human / Kree physiology that rendered her resistant to most toxins and poisons ,
with the added effect of making her virtually invulnerable . Rogue on the cover of Rogue vol 3 # 7 ( March 2005 ) ; art by Frank G. D'Armata , Scot Eaton , and
Don Hillsman Rogue also gained a precognitive `` seventh sense '' that enables her to predict an enemy 's move subconsciously during battle . She used this
ability to predict where Nightcrawler would teleport and from which direction Magus of the Technarchy would attack . This ability was not always reliable ,
however , and would randomly and sporadically activate .
</item 4>

<item 5>
Rogue (comics)
She also still possessed a `` fluid genome '' that enabled Sage to use her as a conduit through which to channel the mutant powers of the X-Treme X-Men team in
a fight against Bogan . Sunfire powers ( edit ) In the short - lived volume three of her self - entitield series ( 2004 -- 2005 ) , Rogue absorbed a large
portion of the mutant Sunfire 's solar - absorption based powers . In addition to her own natural mutant abilities Rogue could then project intense heat and
flame , envelop her body in a fiery aura , fly by focusing her power downwards in a tight stream to propel her like a rocket , focus her power inward to
increase her strength ( though not at her Ms. Marvel levels ) , exercise immunity to heat and radiation , and see the infra - red spectrum . Rogue was
eventually purged of these abilities when she came into contact with Hope Summers . Wonder Man powers ( edit ) Rogue currently possesses Wonder Man 's essence ,
including his powers and psyche . Her organic tissues have been permeated with ionic energy , granting her an array of superhuman abilities .
</item 5>

<item 6>
Rogue (comics)
When Professor Xavier removes the blocks , her powers are allowed to develop normally . With this new control , Rogue demonstrates the ability to absorb and
collectively utilize all the powers of the New X-Men , in order to defeat a rogue Predator X , with no apparent harm to either them or herself . Her power also
affects aliens , such as when she was transferred to a planet in another dimension by Magik , and one of the aliens involuntarily touches her and she gains the
ability to communicate in their language , along with their knowledge . She later uses a more lethal version of her powers against the Avengers during her fight
with them , even making Falcon and She - Hulk immediately unconscious without any negative feedback on her physical self ( besides She - Hulk 's green skin )
but she can not control the minds of the ones she absorbs . She absorbs their powers and renders them unconscious for a long time . The new twist in Rogue 's
powers consists of prompting her victims of her skin - to - skin contact , for a consent on the process .
</item 6>

<item 7>
Rogue (comics)
This ability was not always reliable , however , and would randomly and sporadically activate . When she possessed Carol Danvers ' psyche , her `` double ''
consciousness made her highly resistant to telepathic probes , even those of Charles Xavier , which was said to have been a byproduct of two minds existing in
one body and / or Carol Danvers ' Kree physiology . The Ms. Marvel psyche was eventually torn out of Rogue in a subsequent issue , and it was destroyed by
Magneto . X-Treme abilities ( edit ) During the Maximum Security miniseries , Rogue absorbed an injured Cadre K girl named Z'Cann . The girl held on longer than
necessary , using her mind control to boost the process and because of this ( perhaps because she had absorbed a fluid Skrull physiology ) Rogue mutated further
, gaining the ability to ' recall ' the powers of anyone she had ever absorbed before , though she could not control when or which of these powers surfaced .
For a short time Rogue also had a Skrull appearance , scanned as though she had Skrull DNA , and had their shapeshifting powers , but these soon faded and she
returned to her normal appearance .
</item 7>

<item 8>
Rogue (comics)
She possesses superhuman strength sufficient enough to lift well over 100 tons , as well as some degree of superhuman speed , agility , and reflexes . In
addition , Rogue is virtually invulnerable to damage and is capable of flying at high speeds . Rogue 's eyes glow in darkness and her vision extends somewhat
into the infrared spectrum , allowing her to see in the dark . Her new ionic physiology also grants her a form of immortality , protecting her from age and
disease . Natural skills ( edit ) Rogue has shown a fair knowledge of unarmed combat when in a position where she has to fight in situations where she can not
simply use her powers to knock her foe out . She has also received a degree of training from Professor X to protect herself against even subtle telepathic
attacks , this training allowing her to escape Kobik 's reprogramming of various Avengers and villains to reassemble her teammates against this new threat when
even experienced telepaths like Cable were trapped in the illusion . Rogue 's real name ( edit ) In the Going Rogue arc of her 2004 ongoing series , Rogue 's
real name is finally established as Anna Marie .
</item 8>

<item 9>
Rogue (comics)
Z'Cann uses her telepathy to amplify Rogue 's capacity to assimilate memories , causing her powers to mutate . Henceforth , Rogue is able to `` recall ''
previously absorbed powers , but also finds it increasingly difficult to control which powers manifest when , typically possessing Wolverine 's healing factor
and claws , and Cyclops ' optic blasts along with her usual strength . X-Treme X-Men team ( edit ) Rogue was part of the X-Treme X-Men team led by Storm . The
team 's first mission was in search of Destiny 's Diaries ( which prophesied future events ) . During an invasion of Khan ( an alien conqueror from another
dimension ) of the island nation of Madripoor , Rogue requested that Sage use her power of jump - starting abilities to evolve Rogue to a point where she could
control all of the various powers that she has ever imprinted . Sage agreed , and Rogue became a one - woman army , able to use the powers of anyone she had
absorbed in the past all at once . During Khan 's invasion , Rogue was also confronted by Vargas who was said to be `` a new species altogether '' .
</item 9>

<item 10>
Rogue (comics)
Powers and abilities ( edit ) This section relies too much on references to primary sources . Please improve this section by adding secondary or tertiary
sources . ( March 2015 ) ( Learn how and when to remove this template message ) This section describes a work or element of fiction in a primarily in - universe
style . Please help rewrite it to explain the fiction more clearly and provide non-fictional perspective . ( March 2015 ) ( Learn how and when to remove this
template message ) Due to the nature of her unique mutant powers , Rogue has had several different abilities over the decades . Mutant powers ( edit ) Cover art
of Rogue vol. 3 , # 3 ( September 2004 ) ; art by Rodolfo Migliari . Rogue possesses the mutant ability to absorb the psyche and abilities of another human
being ( or members of some sapient alien races ) through skin contact . Rogue can absorb the memories , knowledge , talents , personality and physical abilities
( whether superhuman or not ) of the person she touches , as well as occasionally duplicating in herself physical characteristics of her victim .
</item 10>
=== END ITEM id=d78bfeec764be522 ===

=== ITEM id=c14aa7f28c23a508 ===
Gold sentences and facts:
Sentence 0: Organization of the Petroleum Exporting Countries ( OPEC ) Flag Headquarters Vienna , Austria Official language English Type International cartel Membership 14 states ( May 2017 ) ( show ) Algeria Angola Ecuador Equatorial Guinea Gabon Iran * Iraq * Kuwait * Libya Nigeria Qatar Saudi Arabia * United Arab Emirates Venezuela * * founding member Leaders Secretary General Mohammed Barkindo Establishment Baghdad , Iraq Statute September 1960 In effect January 1961 Currency Indexed as USD per barrel ( US $ / bbl ) Website OPEC.org
  Fact 0.0: The Organization of the Petroleum Exporting Countries (OPEC) has its headquarters in Vienna, Austria.
  Fact 0.1: OPEC's official language is English.
  Fact 0.2: OPEC is an international cartel.
  Fact 0.3: OPEC has 14 member states as of May 2017.
  Fact 0.4: The OPEC member states listed are Algeria, Angola, Ecuador, Equatorial Guinea, Gabon, Iran, Iraq, Kuwait, Libya, Nigeria, Qatar, Saudi Arabia, the United Arab Emirates, and Venezuela.
  Fact 0.5: Iran, Iraq, Kuwait, Saudi Arabia, and Venezuela are marked with an asterisk, which the table defines as founding member.
  Fact 0.6: The OPEC leader is Secretary General Mohammed Barkindo.
  Fact 0.7: OPEC was established in Baghdad, Iraq.
  Fact 0.8: The OPEC statute is dated September 1960 and was in effect from January 1961.
  Fact 0.9: OPEC's prices are indexed as US dollars per barrel (US$/bbl).
  Fact 0.10: OPEC's website is OPEC.org.
Sentence 1: Organization of the Petroleum Exporting Countries ( OPEC , / ˈoʊpɛk / OH - pek , or OPEP in several other languages ) is an intergovernmental organization of 14 nations as of February 2018 , founded in 1960 in Baghdad by the first five members ( Iran , Iraq , Kuwait , Saudi Arabia , and Venezuela ) , and headquartered since 1965 in Vienna , Austria . As of 2016 , the 14 countries accounted for an estimated 44 percent of global oil production and 73 percent of the world 's `` proven '' oil reserves , giving OPEC a major influence on global oil prices that were previously determined by American - dominated multinational oil companies .
  Fact 1.0: OPEC is an intergovernmental organization of 14 nations as of February 2018.
  Fact 1.1: OPEC is pronounced OH-pek (IPA: ˈoʊpɛk), or OPEP in several other languages.
  Fact 1.2: OPEC was founded in 1960 in Baghdad by its first five members: Iran, Iraq, Kuwait, Saudi Arabia, and Venezuela.
  Fact 1.3: OPEC has been headquartered in Vienna, Austria since 1965.
  Fact 1.4: As of 2016, the 14 OPEC countries accounted for an estimated 44 percent of global oil production.
  Fact 1.5: As of 2016, the 14 OPEC countries held an estimated 73 percent of the world's 'proven' oil reserves.
  Fact 1.6: OPEC has a major influence on global oil prices.
  Fact 1.7: Global oil prices were previously determined by American-dominated multinational oil companies.
Sentence 2: Country Region Membership Years Population ( 2016 est . ) Area ( km ) Oil Production ( bbl / day , 2016 ) Proven Reserves ( bbl , 2016 ) Algeria North Africa 1969 -- 40,606,052 2,381,740 1,348,361 12,200,000,000 Angola Southern Africa 2007 -- 28,813,463 1,246,700 1,769,615 8,423,000,000 Ecuador South America 1973 -- 1992 , 2007 -- 16,385,068 283,560 548,421 8,273,000,000 Equatorial Guinea Central Africa 2017 -- 1,221,490 28,051 227,000 1,100,000,000 Gabon Central Africa 1975 -- 1995 , 2016 -- 1,979,786 267,667 210,820 2,000,000,000 Iran Middle East 1960 -- 80,277,428 1,648,000 3,990,956 157,530,000,000 Iraq Middle East 1960 -- 37,202,572 437,072 4,451,516 143,069,000,000 Kuwait Middle East 1960 -- 4,052,584 17,820 2,923,825 101,500,000,000 Libya North Africa 1962 -- 6,293,253 1,759,540 384,686 48,363,000,000 Nigeria West Africa 1971 -- 185,989,640 923,768 1,999,885 37,070,000,000 Qatar Middle East 1961 -- 2,569,804 11,437 1,522,902 25,244,000,000 Saudi Arabia Middle East 1960 -- 32,275,687 2,149,690 10,460,710 266,578,000,000 United Arab Emirates Middle East 1967 -- 9,269,612 83,600 3,106,077 97,800,000,000 Venezuela South America 1960 -- 31,568,179 912,050 2,276,967 299,953,000,000 OPEC Total 478,498,000 12,150,695 35,221,740 1,209,103,000,000 World Total 7,618,511,000 510,072,000 80,622,287 1,650,585,000,000 OPEC Percent 6 % 2 % 44 % 73 %
  Fact 2.0: Algeria (North Africa): membership from 1969; population 40,606,052 (2016 est.); area 2,381,740 km²; oil production 1,348,361 bbl/day (2016); proven reserves 12,200,000,000 bbl (2016).
  Fact 2.1: Angola (Southern Africa): membership from 2007; population 28,813,463; area 1,246,700 km²; oil production 1,769,615 bbl/day; proven reserves 8,423,000,000 bbl.
  Fact 2.2: Ecuador (South America): membership 1973 to 1992 and from 2007; population 16,385,068; area 283,560 km²; oil production 548,421 bbl/day; proven reserves 8,273,000,000 bbl.
  Fact 2.3: Equatorial Guinea (Central Africa): membership from 2017; population 1,221,490; area 28,051 km²; oil production 227,000 bbl/day; proven reserves 1,100,000,000 bbl.
  Fact 2.4: Gabon (Central Africa): membership 1975 to 1995 and from 2016; population 1,979,786; area 267,667 km²; oil production 210,820 bbl/day; proven reserves 2,000,000,000 bbl.
  Fact 2.5: Iran (Middle East): membership from 1960; population 80,277,428; area 1,648,000 km²; oil production 3,990,956 bbl/day; proven reserves 157,530,000,000 bbl.
  Fact 2.6: Iraq (Middle East): membership from 1960; population 37,202,572; area 437,072 km²; oil production 4,451,516 bbl/day; proven reserves 143,069,000,000 bbl.
  Fact 2.7: Kuwait (Middle East): membership from 1960; population 4,052,584; area 17,820 km²; oil production 2,923,825 bbl/day; proven reserves 101,500,000,000 bbl.
  Fact 2.8: Libya (North Africa): membership from 1962; population 6,293,253; area 1,759,540 km²; oil production 384,686 bbl/day; proven reserves 48,363,000,000 bbl.
  Fact 2.9: Nigeria (West Africa): membership from 1971; population 185,989,640; area 923,768 km²; oil production 1,999,885 bbl/day; proven reserves 37,070,000,000 bbl.
  Fact 2.10: Qatar (Middle East): membership from 1961; population 2,569,804; area 11,437 km²; oil production 1,522,902 bbl/day; proven reserves 25,244,000,000 bbl.
  Fact 2.11: Saudi Arabia (Middle East): membership from 1960; population 32,275,687; area 2,149,690 km²; oil production 10,460,710 bbl/day; proven reserves 266,578,000,000 bbl.
  Fact 2.12: United Arab Emirates (Middle East): membership from 1967; population 9,269,612; area 83,600 km²; oil production 3,106,077 bbl/day; proven reserves 97,800,000,000 bbl.
  Fact 2.13: Venezuela (South America): membership from 1960; population 31,568,179; area 912,050 km²; oil production 2,276,967 bbl/day; proven reserves 299,953,000,000 bbl.
  Fact 2.14: OPEC total: population 478,498,000; area 12,150,695 km²; oil production 35,221,740 bbl/day; proven reserves 1,209,103,000,000 bbl.
  Fact 2.15: World total: population 7,618,511,000; area 510,072,000 km²; oil production 80,622,287 bbl/day; proven reserves 1,650,585,000,000 bbl.
  Fact 2.16: OPEC percent of the world: population 6%; area 2%; oil production 44%; proven reserves 73%.

Retrieved items:
<item 1>
OPEC
269,612 83,600 3,106,077 97,800,000,000 Venezuela South America 1960 -- 31,568,179 912,050 2,276,967 299,953,000,000 OPEC Total 478,498,000 12,150,695
35,221,740 1,209,103,000,000 World Total 7,618,511,000 510,072,000 80,622,287 1,650,585,000,000 OPEC Percent 6 % 2 % 44 % 73 % ^ Jump up to : One petroleum
barrel ( bbl ) is approximately 42 U.S. gallons , or 159 liters , or 0.159 m , varying slightly with temperature . To put the production numbers in context , a
supertanker typically holds 2,000,000 barrels ( 320,000 m ) , and the world 's current production rate would take approximately 56 years to exhaust the world 's
current proven reserves . ^ Jump up to : The five founding members attended the first OPEC conference in September 1960 .
</item 1>

<item 2>
OPEC
9 2003 -- 2011 volatility 3.10 2008 production dispute 3.11 2014 -- 2017 oil glut 3.12 2017 - 2018 Production cut 4 See also 5 References 6 External links
Membership ( edit ) Current member countries ( edit ) As of May 2017 , OPEC has 14 member countries : six in the Middle East ( Western Asia ) , six in Africa ,
and two in South America . According to the U.S. Energy Information Administration ( EIA ) , OPEC 's combined rate of oil production ( including gas condensate
) represented 44 percent of the world 's total in 2016 , and OPEC accounted for 73 percent of the world 's `` proven '' oil reserves , including 48 percent from
just the six Middle Eastern members : Approval of a new member country requires agreement by three - quarters of OPEC 's existing members , including all five
of the founders . In October 2015 , Sudan formally submitted an application to join , but it is not yet a member . Country Region Membership Years Population (
2016 est . )
</item 2>

<item 3>
OPEC
During 1961 -- 1975 , the five founding nations were joined by Qatar ( 1961 ) , Indonesia ( 1962 -- 2008 , rejoined 2014 - 2016 ) , Libya ( 1962 ) , United Arab
Emirates ( originally just the Emirate of Abu Dhabi , 1967 ) , Algeria ( 1969 ) , Nigeria ( 1971 ) , Ecuador ( 1973 -- 1992 , rejoined 2007 ) , and Gabon ( 1975
-- 1994 , rejoined 2016 ) . By the early 1970s , OPEC 's membership accounted for more than half of worldwide oil production . Indicating that OPEC is not
averse to further expansion , Mohammed Barkindo , OPEC 's Acting Secretary General in 2006 , urged his African neighbors Angola and Sudan to join , and Angola
did in 2007 , followed by Equatorial Guinea in 2017 . Since the 1980s , representatives from Egypt , Mexico , Norway , Oman , Russia , and other oil - exporting
nations have attended many OPEC meetings as observers , as an informal mechanism for coordinating policies .
</item 3>

<item 4>
OPEC
Bloomberg News . 11 December 2016 . Jump up ^ `` OPEC Said to Break Bread With Shale in Rare Show of Detente '' . Bloomberg News . 7 March 2017 . Jump up ^ ``
Russia backs gradual , managed exit from oil cuts with OPEC '' . Reuters. 2017 - 12 - 22 . Retrieved 2017 - 12 - 25 . External links ( edit ) Energy portal
Wikimedia Commons has media related to OPEC . Official website The OPEC Fund for International Development ( OFID ) official website ( hide ) Organization of
the Petroleum Exporting Countries ( OPEC ) Algeria Angola Ecuador Equatorial Guinea Gabon Iran Iraq Kuwait Libya Nigeria Qatar Saudi Arabia United Arab Emirates
Venezuela List of Secretaries General OPEC Fund for International Development OPEC Reference Basket Heads of state of OPEC member states Bouteflika Lourenço
Moreno Obiang Bongo Khamenei Masum Sabah Sarraj Buhari Tamim Salman Khalifa Maduro Acting heads of state shown in italics . Petroleum industry Petroleum Primary
energy Benchmarks Argus Sour Bonny Light Brent Dubai Indian Basket Indonesian Isthmus - 34 Light Japan Cocktail OPEC
</item 4>

<item 5>
OPEC
the petroleum industry Nationalization OPEC GECF Seven Sisters Standard Oil Oil market timelines Provinces and fields List of natural gas fields List of oil
fields East Midlands Oil Province East Texas Gulf of Mexico Niger Delta North Sea Permian Basin Persian Gulf Prudhoe Bay Oil Field Russia Venezuela Western
Canadian Sedimentary Basin Other topics Acronyms Oil shale gas Peak oil mitigation timing People Petrocurrency Petrodollar recycling Shale band Shale gas Swing
producer Unconventional oil heavy crude oil sands oil shale Companies and organisations Major petroleum companies Supermajors BP Chevron Eni ExxonMobil Royal
Dutch Shell Total National oil companies ADNOC ( UAE ) CNOOC ( China ) CNPC ( China ) Ecopetrol ( Colombia ) Gazprom ( Russia ) Iraq National Oil Company Indian
Oil Corporation KazMunayGas ( Kazakhstan ) Kuwait Petroleum Corporation Lotos ( Poland ) Nigerian National Petroleum Corporation NIOC ( Iran ) NISOC ( Iran )
OGDCL ( Pakistan ) ONGC ( India ) PDVSA ( Venezuela ) PKN Orlen ( Poland ) Pemex ( Mexico ) Pertamina ( Indonesia ) PetroBangla ( Bangladesh ) Petrobras (
Brazil ) PetroChina Petronas ( Malaysia ) Petrovietnam PTT ( Thailand )
</item 5>

<item 6>
OPEC
OPEC - wikipedia OPEC Jump to : navigation , search Organization of the Petroleum Exporting Countries ( OPEC ) Flag Headquarters Vienna , Austria Official
language English Type International cartel Membership 14 states ( May 2017 ) ( show ) Algeria Angola Ecuador Equatorial Guinea Gabon Iran * Iraq * Kuwait *
Libya Nigeria Qatar Saudi Arabia * United Arab Emirates Venezuela * * founding member Leaders Secretary General Mohammed Barkindo Establishment Baghdad , Iraq
Statute September 1960 In effect January 1961 Currency Indexed as USD per barrel ( US $ / bbl ) Website OPEC.org Organization of the Petroleum Exporting
Countries ( OPEC , / ˈoʊpɛk / OH - pek , or OPEP in several other languages ) is an intergovernmental organization of 14 nations as of February 2018 , founded
in 1960 in Baghdad by the first five members ( Iran , Iraq , Kuwait , Saudi Arabia , and Venezuela ) , and headquartered since 1965 in Vienna , Austria .
</item 6>

<item 7>
OPEC
As of 2016 , the 14 countries accounted for an estimated 44 percent of global oil production and 73 percent of the world 's `` proven '' oil reserves , giving
OPEC a major influence on global oil prices that were previously determined by American - dominated multinational oil companies . OPEC 's stated mission is ``
to coordinate and unify the petroleum policies of its member countries and ensure the stabilization of oil markets , in order to secure an efficient , economic
and regular supply of petroleum to consumers , a steady income to producers , and a fair return on capital for those investing in the petroleum industry . ''
The organization is also a significant provider of information about the international oil market . As of May 2017 , OPEC 's members are Algeria , Angola ,
Ecuador , Equatorial Guinea , Gabon , Iran , Iraq , Kuwait , Libya , Nigeria , Qatar , Saudi Arabia ( the de facto leader ) , United Arab Emirates , and
Venezuela , while Indonesia is a former member . Two - thirds of OPEC 's oil production and reserves are in its six Middle Eastern countries that surround the
oil - rich Persian Gulf .
</item 7>

<item 8>
OPEC
However , their influence on international trade is periodically challenged by the expansion of non-OPEC energy sources , and by the recurring temptation for
individual OPEC countries to exceed production ceilings and pursue conflicting self - interests . Contents ( hide ) 1 Membership 1.1 Current member countries
1.2 Lapsed members 1.3 Observers 1.4 Leadership and decision - making 1.5 International cartel 1.6 Conflicts 2 Market information 2.1 Publications and research
2.2 Crude oil benchmarks 2.3 Spare capacity 3 History and impact 3.1 Post-WWII situation 3.2 1959 -- 1960 anger from exporting countries 3.3 1960 -- 1975
founding and expansion 3.4 1973 -- 1974 oil embargo 3.5 1975 -- 1980 Special Fund , now OFID 3.6 1975 hostage siege 3.7 1979 -- 1980 oil crisis and 1980s oil
glut 3.8 1990 -- 2003 ample supply and modest disruptions 3.9 2003 -- 2011 volatility 3.
</item 8>

<item 9>
OPEC
In May 2008 , Indonesia announced that it would leave OPEC when its membership expired at the end of that year , having become a net importer of oil and being
unable to meet its production quota . It rejoined the organization in January 2016 , but announced another `` temporary suspension '' of its membership at year
- end when OPEC requested a 5 percent production cut . Some commentators consider that the United States was a de facto member of OPEC during its formal
occupation of Iraq , due to its leadership of the Coalition Provisional Authority in 2003 -- 2004 . But this is not borne out by the minutes of OPEC meetings ,
as no US representative attended in an official capacity . Observers ( edit ) Since the 1980s , representatives from Egypt , Mexico , Norway , Oman , Russia ,
and other oil - exporting nations have attended many OPEC meetings as observers . This arrangement serves as an informal mechanism for coordinating policies .
</item 9>

<item 10>
OPEC
History and impact ( edit ) Post-WWII situation ( edit ) In 1949 , Venezuela and Iran took the earliest steps in the direction of OPEC , by inviting Iraq ,
Kuwait and Saudi Arabia to improve communication among petroleum - exporting nations as the world recovered from World War II . At the time , some of the world
's largest oil fields were just entering production in the Middle East . The United States had established the Interstate Oil Compact Commission to join the
Texas Railroad Commission in limiting overproduction . The US was simultaneously the world 's largest producer and consumer of oil ; and the world market was
dominated by a group of multinational companies known as the `` Seven Sisters '' , five of which were headquartered in the US following the breakup of John D.
Rockefeller 's original Standard Oil monopoly . Oil - exporting countries were eventually motivated to form OPEC as a counterweight to this concentration of
political and economic power . 1959 -- 1960 anger from exporting countries ( edit ) In February 1959 , as new supplies were becoming available , the
multinational oil companies ( MOCs ) unilaterally reduced their posted prices for Venezuelan and Middle Eastern crude oil by 10 percent .
</item 10>
=== END ITEM id=c14aa7f28c23a508 ===

=== ITEM id=b9b83a18c3d600d4 ===
Gold sentences and facts:
Sentence 0: An elevator with a counterbalance approximates an ideal Atwood machine and thereby relieves the driving motor from the load of holding the elevator cab -- it has to overcome only weight difference and inertia of the two masses . The same principle is used for funicular railways with two connected railway cars on inclined tracks , and for the elevators on the Eiffel Tower which counterbalance each other . Ski lifts are another example , where the gondolas move on a closed ( continuous ) pulley system up and down the mountain . The ski lift is similar to the counter-weighted elevator , but with a constraining force provided by the cable in the vertical dimension thereby achieving work in both the horizontal and vertical dimensions . Boat lifts are another type of counter-weighted elevator system approximating an Atwood machine .
  Fact 0.0: An elevator with a counterbalance approximates an ideal Atwood machine.
  Fact 0.1: A counterbalanced elevator relieves the driving motor from the load of holding the elevator cab.
  Fact 0.2: A counterbalanced elevator's driving motor has to overcome only the weight difference and inertia of the two masses.
  Fact 0.3: The same principle is used for funicular railways with two connected railway cars on inclined tracks.
  Fact 0.4: The same principle is used for the elevators on the Eiffel Tower, which counterbalance each other.
  Fact 0.5: Ski lifts are another example.
  Fact 0.6: In ski lifts, the gondolas move on a closed (continuous) pulley system up and down the mountain.
  Fact 0.7: A ski lift is similar to the counter-weighted elevator, but with a constraining force provided by the cable in the vertical dimension.
  Fact 0.8: A ski lift achieves work in both the horizontal and vertical dimensions.
  Fact 0.9: Boat lifts are another type of counter-weighted elevator system approximating an Atwood machine.

Retrieved items:
<item 1>
Atwood machine
) + ( ( I ) \ over ( r ^ ( 2 ) ) ) ) ) ) Tension in string segment nearest m : T 1 = m 1 g ( 2 m 2 + I r 2 ) m 1 + m 2 + I r 2 ( \ displaystyle T_ ( 1 ) = ( (
m_ ( 1 ) g ( 2m_ ( 2 ) + ( ( I ) \ over ( r ^ ( 2 ) ) ) ) ) \ over ( m_ ( 1 ) + m_ ( 2 ) + ( ( I ) \ over ( r ^ ( 2 ) ) ) ) ) ) Tension in string segment
nearest m : T 2 = m 2 g ( 2 m 1 + I r 2 ) m 1 + m 2 + I r 2 ( \ displaystyle T_ ( 2 ) = ( ( m_ ( 2 ) g ( 2m_ ( 1 ) + ( ( I ) \ over ( r ^ ( 2 ) ) ) ) ) \ over (
m_ ( 1 ) + m_ ( 2 ) + ( ( I ) \
</item 1>

<item 2>
Atwood machine
_ ( \ mathrm ( friction ) ) \ over r ) ) \ over ( m_ ( 1 ) + m_ ( 2 ) + ( ( I ) \ over ( r ^ ( 2 ) ) ) ) ) ) Tension in string segment nearest m : T 1 = m 1 g (
2 m 2 + I r 2 + τ f r i c t i o n r g ) m 1 + m 2 + I r 2 ( \ displaystyle T_ ( 1 ) = ( ( m_ ( 1 ) g ( 2m_ ( 2 ) + ( ( I ) \ over ( r ^ ( 2 ) ) ) + ( ( \ tau _
( \ mathrm ( friction ) ) ) \ over ( rg ) ) ) ) \ over ( m_ ( 1 ) + m_ ( 2 ) + ( ( I ) \ over ( r ^ ( 2 ) ) ) ) ) ) Tension in string segment nearest m : T 2 =
m 2 g ( 2 m 1 + I r 2 + τ f r i c t i o n r g ) m 1 + m 2
</item 2>

<item 3>
Atwood machine
+ τ f r i c t i o n r g ) m 1 + m 2 + I r 2 ( \ displaystyle T_ ( 2 ) = ( ( m_ ( 2 ) g ( 2m_ ( 1 ) + ( ( I ) \ over ( r ^ ( 2 ) ) ) + ( ( \ tau _ ( \ mathrm (
friction ) ) ) \ over ( rg ) ) ) ) \ over ( m_ ( 1 ) + m_ ( 2 ) + ( ( I ) \ over ( r ^ ( 2 ) ) ) ) ) ) Should bearing friction be negligible ( but not the
inertia of the pulley and not the traction of the string on the pulley rim ) , these equations simplify as the following results : Acceleration : a = g ( m 1 −
m 2 ) m 1 + m 2 + I r 2 ( \ displaystyle a = ( ( g ( m_ ( 1 ) - m_ ( 2 ) ) ) \ over ( m_ ( 1 ) + m_ ( 2 ) + ( ( I ) \ over ( r ^ ( 2 ) ) ) ) ) )
</item 3>

<item 4>
Atwood machine
Boat lifts are another type of counter-weighted elevator system approximating an Atwood machine . See also ( edit ) Wikimedia Commons has media related to
Atwood 's machine . Frictionless plane Kater 's pendulum Spherical cow Swinging Atwood 's machine Notes ( edit ) Jump up ^ Tipler , Paul A. ( 1991 ) . Physics
For Scientists and Engineers ( 3rd , extended ed . ) . New York : Worth Publishers . p. 160 . ISBN 0 - 87901 - 432 - 6 . Chapter 6 , example 6 - 13 Jump up ^
Goldstein , Herbert ( 1980 ) . Classical Mechanics ( 2nd ed . ) . New Delhi : Addison - Wesley / Narosa Indian Student Edition . pp. 26 -- 27 . ISBN 81 - 85015
- 53 - 8 .
</item 4>

<item 5>
Atwood machine
The angular acceleration of the pulley is given by the no - slip condition : α = a r , ( \ displaystyle \ alpha = ( a \ over r ) , ) where α ( \ displaystyle \
alpha ) is the angular acceleration . The net torque is then : τ n e t = ( T 1 − T 2 ) r − τ f r i c t i o n = I α ( \ displaystyle \ tau _ ( \ mathrm ( net ) )
= \ left ( T_ ( 1 ) - T_ ( 2 ) \ right ) r - \ tau _ ( \ mathrm ( friction ) ) = I \ alpha ) Combining with Newton 's second law for the hanging masses , and
solving for T , T , and a , we get : Acceleration : a = g ( m 1 − m 2 ) − τ f r i c t i o n r m 1 + m 2 + I r 2 ( \ displaystyle a = ( ( g ( m_ ( 1 ) - m_ ( 2 )
) - ( \ tau _ ( \ mathrm ( friction ) ) \ over r ) ) \ over ( m_ (
</item 5>

<item 6>
Atwood machine
( m_ ( 1 ) + m_ ( 2 ) + ( ( I ) \ over ( r ^ ( 2 ) ) ) ) ) ) Practical implementations ( edit ) Atwood 's original illustrations show the main pulley 's axle
resting on the rims of another four wheels , to minimize friction forces from the bearings . Many historical implementations of the machine follow this design .
An elevator with a counterbalance approximates an ideal Atwood machine and thereby relieves the driving motor from the load of holding the elevator cab -- it
has to overcome only weight difference and inertia of the two masses . The same principle is used for funicular railways with two connected railway cars on
inclined tracks , and for the elevators on the Eiffel Tower which counterbalance each other . Ski lifts are another example , where the gondolas move on a
closed ( continuous ) pulley system up and down the mountain . The ski lift is similar to the counter-weighted elevator , but with a constraining force provided
by the cable in the vertical dimension thereby achieving work in both the horizontal and vertical dimensions . Boat lifts are another type of counter-weighted
elevator system approximating an Atwood machine .
</item 6>

<item 7>
Atwood machine
ISBN 81 - 85015 - 53 - 8 . Section 1 - 6 , example 2 Jump up ^ See Wikipedia article : https://en.wikipedia.org/wiki/Boat_lift External links ( edit ) Professor
Greenslade 's account on the Atwood Machine Atwood 's Machine by Enrique Zeleny , The Wolfram Demonstrations Project . Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Atwood_machine&oldid=792136529 '' Categories : Mechanics Physics experiments Talk Contents About Wikipedia Català
Deutsch Español Français 한국어 Hrvatski Italiano Nederlands 日本 語 Polski Português Русский Srpskohrvatski / српскохрватски Türkçe Українська 中文 Edit links This
page was last edited on 24 July 2017 , at 17 : 58 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply .
By using this site , you agree to the Terms of Use and Privacy Policy .
</item 7>

<item 8>
Atwood machine
By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit
organization . About Wikipedia
</item 8>

<item 9>
Atwood machine
Weight of m 1 ( \ displaystyle m_ ( 1 ) ) and m 2 ( \ displaystyle m_ ( 2 ) ) is simply W 1 = m 1 g ( \ displaystyle W_ ( 1 ) = m_ ( 1 ) g ) and W 2 = m 2 g ( \
displaystyle W_ ( 2 ) = m_ ( 2 ) g ) respectively . Forces affecting m : m 1 g − T = m 1 a ( \ displaystyle \ ; m_ ( 1 ) g-T = m_ ( 1 ) a ) Forces affecting m :
T − m 2 g = m 2 a ( \ displaystyle \ ; T - m_ ( 2 ) g = m_ ( 2 ) a ) and adding the two previous equations we obtain m 1 g − m 2 g = m 1 a + m 2 a ( \
displaystyle \ ; m_ ( 1 ) g - m_ ( 2 ) g = m_ ( 1 ) a + m_ ( 2 ) a ) ,
</item 9>

<item 10>
Atwood machine
Atwood machine - wikipedia Atwood machine Jump to : navigation , search Illustration of Atwood machine , 1905 . The Atwood machine ( or Atwood 's machine ) was
invented in 1784 by the English mathematician George Atwood as a laboratory experiment to verify the mechanical laws of motion with constant acceleration .
Atwood 's machine is a common classroom demonstration used to illustrate principles of classical mechanics . The ideal Atwood Machine consists of two objects of
mass m and m , connected by an inextensible massless string over an ideal massless pulley . When m = m , the machine is in neutral equilibrium regardless of the
position of the weights . When m ≠ m both masses experience uniform acceleration . Contents ( hide ) 1 Equation for constant acceleration 2 Equation for tension
3 Equations for a pulley with inertia and friction 3.1 Practical implementations 4 See also 5 Notes 6 External links Equation for constant acceleration ( edit )
The free body diagrams of the two hanging masses of the Atwood machine .
</item 10>
=== END ITEM id=b9b83a18c3d600d4 ===

=== ITEM id=d928f5318e713c14 ===
Gold sentences and facts:
Sentence 0: Tracy McConnell , better known as `` The Mother '' , is the title character from the CBS television sitcom How I Met Your Mother . The show , narrated by Future Ted , tells the story of how Ted Mosby met The Mother . Tracy McConnell appears in 8 episodes from `` Lucky Penny '' to `` The Time Travelers '' as an unseen character ; she was first seen fully in `` Something New '' and was promoted to a main character in season 9 . The Mother is played by Cristin Milioti .
  Fact 0.0: Tracy McConnell is better known as 'The Mother'.
  Fact 0.1: Tracy McConnell is the title character from the CBS television sitcom How I Met Your Mother.
  Fact 0.2: The sitcom is narrated by Future Ted.
  Fact 0.3: The sitcom tells the story of how Ted Mosby met The Mother.
  Fact 0.4: Tracy McConnell appears in 8 episodes.
  Fact 0.5: Tracy McConnell's episodes run from 'Lucky Penny' to 'The Time Travelers'.
  Fact 0.6: Tracy McConnell appears as an unseen character in those episodes.
  Fact 0.7: Tracy McConnell was first seen fully in 'Something New'.
  Fact 0.8: Tracy McConnell was promoted to a main character in season 9.
  Fact 0.9: The Mother is played by Cristin Milioti.

Retrieved items:
<item 1>
The Mother (How I Met Your Mother)
In the season 8 episode `` Band or DJ ? , '' Ted runs into Cindy on the subway and tells her that the band Barney and Robin hired to play at their wedding
cancelled at the last minute . The end result of the encounter is that Cindy 's ( now ex - ) roommate 's band plays at Barney and Robin 's wedding . The Mother
is first shown meeting Louis in `` How Your Mother Met Me '' as she is left to carry the band equipment while the now - lead band member Darren talks to his
fans . Later at MacLaren 's Pub , she tells him she 's not yet ready to date . Louis asks her to give him a call if she changes her mind and they begin dating
not long after . The Mother meets all of Ted 's best friends ( Barney , Lily , Marshall and Robin ) before she meets him . The Mother is responsible for
convincing Barney to pursue Robin , as revealed through a flashback in `` Platonish '' . In `` The Locket '' , Tracy meets Lily on a train journey .
</item 1>

<item 2>
The Mother (How I Met Your Mother)
After they start talking , Mitch encourages her to pursue her dreams . The Mother expresses her desire to end poverty by taking up economics in college . On his
first day of teaching as Professor Mosby , as seen in the season 4 finale `` The Leap '' , he is seen in front of the classroom of students , one of which
Future Ted says is the titular mother . But in the first episode of season 5 , `` Definitions '' , it is revealed that he was actually in the wrong classroom --
Economics instead of Architecture . At the same time in `` How Your Mother Met Me '' , the Mother sits her first session in Economics 305 and meets another
graduate student named Cindy ( Rachel Bilson ) , whom she offers to move in with her as her roommate . They see Ted enter the room , but when he announces the
subject , The Mother thinks she is in the wrong room and runs off . She heads back to the room after seeing Ted scramble to his actual classroom . Later , in ``
Girls Versus Suits '' , Ted dates Cindy , not knowing that her roommate is his future wife .
</item 2>

<item 3>
The Mother (How I Met Your Mother)
In the video , future Ted is heard saying , `` ... When I think how lucky I am to wake up next to your mom every morning , I ca n't help but be amazed how easy
it all really was ... '' , indirectly stating that The Mother is alive . The video ends right after the train passes at Farhampton station and credits start
rolling , implying that Ted never went back to Robin as he lived a successful married life with Tracy Mosby . Contents ( hide ) 1 Casting 2 Character history
2.1 Name 3 Death 4 Notes 5 References Casting ( edit ) During its first eight seasons , the successful sitcom How I Met Your Mother often hinted at the unseen
character of The Mother . Well - known actresses often made guest appearances on the show . Many fans expected that another would play one of the most - wanted
roles in Hollywood , but creators Carter Bays and Craig Thomas wanted an unknown .
</item 3>

<item 4>
The Mother (How I Met Your Mother)
Rolling Stone . Retrieved November 16 , 2014 . ( hide ) How I Met Your Mother Episodes Season 1 Season 2 Season 3 Season 4 Season 5 Season 6 Season 7 Season 8
Season 9 Characters Ted Mosby Marshall Eriksen Lily Aldrin Barney Stinson Robin Scherbatsky The Mother ( Tracy McConnell ) Soundtracks How I Met Your Music
Related articles Awards and nominations The Bro Code Cheerleader effect `` Nothing Suits Me Like a Suit '' Retrieved from ``
https://en.wikipedia.org/w/index.php?title=The_Mother_(How_I_Met_Your_Mother)&oldid=802354471 '' Categories : How I Met Your Mother characters Fictional
characters introduced in 2013 Fictional female musicians Hidden categories : Pages using deprecated image syntax Pages using infobox character with unknown
parameters Talk Contents About Wikipedia Español Italiano Magyar Tiếng Việt Edit links This page was last edited on 25 September 2017 , at 17 : 03 . Text is
available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply .
</item 4>

<item 5>
The Mother (How I Met Your Mother)
Tracy McConnell appears in 8 episodes from `` Lucky Penny '' to `` The Time Travelers '' as an unseen character ; she was first seen fully in `` Something New
'' and was promoted to a main character in season 9 . The Mother is played by Cristin Milioti . The story of how Ted met The Mother is the framing device behind
the series ; many facts about her are revealed throughout the series , including the fact that Ted once unwittingly owned her umbrella before accidentally
leaving it behind in her apartment . Ted and The Mother meet at the Farhampton train station following Barney Stinson and Robin Scherbatsky 's wedding ; this
scene is shown in `` Last Forever '' , the series finale . The Mother 's death from an unspecified terminal illness in 2024 , also revealed in the series finale
, received a mixed reaction from fans . An alternate ending was released in the ninth season DVD . In the alternate ending , Tracy Mosby is still living when
Ted is telling the story in 2030 .
</item 5>

<item 6>
The Mother (How I Met Your Mother)
The Mother ( How I Met Your Mother ) - wikipedia The Mother ( How I Met Your Mother ) Jump to : navigation , search Tracy McConnell How I Met Your Mother
character The Mother appearing in `` The Locket '' First appearance `` Lucky Penny ( unseen ) '' `` Something New '' ( seen ) Last appearance `` Last Forever ''
Created by Carter Bays Craig Thomas Portrayed by Cristin Milioti Information Aliases The Mother Gender Female Spouse ( s ) Ted Mosby Significant other ( s ) Max
( deceased former boyfriend ) Louis ( ex-boyfriend ) Children Penny Mosby ( daughter , born in 2015 , played by Lyndsy Fonseca ) Luke Mosby ( son , born in 2017
, played by David Henrie ) Nationality American Tracy McConnell , better known as `` The Mother '' , is the title character from the CBS television sitcom How I
Met Your Mother . The show , narrated by Future Ted , tells the story of how Ted Mosby met The Mother .
</item 6>

<item 7>
The Mother (How I Met Your Mother)
In a flashforward in `` The Lighthouse '' , Ted proposes to the Mother at the top of the lighthouse near Farhampton Inn . She immediately accepts . In another
flashfoward in `` Unpause '' , the Mother is revealed to be pregnant with their second child , Luke , in the year 2017 . She goes into labor while she and Ted
are staying at Farhampton . Name ( edit ) The Mother 's real name is not revealed until the series finale , `` Last Forever '' . When Ted meets her at the
Farhampton train station , she reveals that her name is Tracy McConnell . In the season 1 episode `` Belly Full of Turkey '' , Ted meets a stripper named Tracy
and says `` ... that , kids , is the true story of how I met your mother '' . The children react in surprise and appear to believe Ted before he admits he is
joking , which led some fans to correctly guess that The Mother 's name is Tracy .
</item 7>

<item 8>
The Mother (How I Met Your Mother)
Using Anne Hathaway and Amy Adams as examples , Thomas said that `` We did n't want it to be a big famous star because we did n't want the wider audience to
have associations with whatever actress this would be ... The whole idea is that Ted 's never seen this woman before , so it better feel that way to the
audience '' , similar to how Cobie Smulders being cast as Robin Scherbatsky had `` kept the show alive '' when it began . Bays and Thomas also did not want a
large casting call . They chose Cristin Milioti after seeing her on 30 Rock and Once ; her musical ability was also helpful , as The Mother had been described
as a band member . After competing for the role against at least two others , Milioti filmed her first scene -- for the last episode of season 8 -- having never
watched How I Met Your Mother ; she recalled , `` I had ignorance on my side . So I did n't know what it meant . '' Milioti learned of the character 's
importance only after binge watching the show during the summer .
</item 8>

<item 9>
The Mother (How I Met Your Mother)
While this incident made Cindy realise that she is a lesbian , it also made the Mother decide to go back into dating , as the kiss was her first in a long time
. Some time after this , a man named Darren approaches The Mother and is welcomed into her band named Superfreakonomics . Darren gradually takes over the band .
In the season 6 opener `` Big Days '' it is revealed Ted meets his future wife `` the day of '' the wedding at which he is the best man . In the episode ``
False Positive '' Robin asks Ted to be her future best man , should she ever get married . In the episode `` Challenge Accepted '' , it is revealed that Ted
meets the mother of his children the day of Barney 's wedding . In the last episode of season 7 , `` The Magician 's Code '' it is shown that Barney will marry
Robin , and Ted will meet the mother `` the day of '' their wedding . On the premiere of season 8 , Ted 's wife appears after Barney and Robin 's wedding ,
outside at the `` Farhampton '' station while holding a yellow umbrella and her bass guitar .
</item 9>

<item 10>
Graph query row:
person: Cristin Milioti
relationship: RELATED_TO
role: None
relationship_detail: None
relation: None
context: None
fact: None
character: Tracy McConnell
</item 10>
=== END ITEM id=d928f5318e713c14 ===
