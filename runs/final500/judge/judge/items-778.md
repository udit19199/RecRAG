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

=== ITEM id=427551a8f2170b45 ===
Gold sentences and facts:
Sentence 0: Loyalists were American colonists who remained loyal to the British Crown during the American Revolutionary War . At the time they were often called Tories , Royalists , or King 's Men . They were opposed by the Patriots , those who supported the revolution and called them `` persons inimical to the liberties of America '' . Prominent Loyalists repeatedly assured the British government that many thousands of loyalists would spring to arms and fight for the crown . The British government acted in expectation of that , especially in the southern campaigns in 1780 - 81 . In practice , the number of loyalists in military service was far lower than expected . Across the colonies , Patriots watched suspected Loyalists very closely , and would not tolerate any organized Loyalist opposition . Many outspoken or militarily active loyalists were forced to flee , especially to their stronghold of New York City . William Franklin , the royal governor of New Jersey and son of Patriot leader Benjamin Franklin , became the leader of the Loyalists after his release from a Patriot prison in 1778 . He worked to build Loyalist military units to fight in the war , but the number of volunteers was much fewer than London expected .
  Fact 0.0: Loyalists were American colonists who remained loyal to the British Crown during the American Revolutionary War.
  Fact 0.1: Loyalists were called Tories, Royalists, or King's Men at the time.
  Fact 0.2: Loyalists were opposed by the Patriots.
  Fact 0.3: The Patriots were those who supported the revolution.
  Fact 0.4: The Patriots called the Loyalists "persons inimical to the liberties of America."
  Fact 0.5: Prominent Loyalists repeatedly assured the British government that many thousands of loyalists would spring to arms and fight for the Crown.
  Fact 0.6: The British government acted in expectation of that, especially in the southern campaigns in 1780-81.
  Fact 0.7: In practice, the number of loyalists in military service was far lower than expected.
  Fact 0.8: Patriots watched suspected Loyalists very closely across the colonies.
  Fact 0.9: Patriots would not tolerate any organized Loyalist opposition.
  Fact 0.10: Many outspoken or militarily active loyalists were forced to flee.
  Fact 0.11: Loyalists fled especially to their stronghold of New York City.
  Fact 0.12: William Franklin was the royal governor of New Jersey.
  Fact 0.13: William Franklin was the son of Patriot leader Benjamin Franklin.
  Fact 0.14: William Franklin became the leader of the Loyalists after his release from a Patriot prison in 1778.
  Fact 0.15: William Franklin worked to build Loyalist military units to fight in the war.
  Fact 0.16: The number of volunteers for William Franklin's Loyalist military units was much fewer than London expected.

Retrieved items:
<item 1>
Loyalist (American Revolution)
1778 ) , led a band of Tories , commissioned by Lord Dunmore , who upon capture was executed John Randolph ( 1727 -- 1784 ) , King 's Attorney for Virginia Anna
Rawle , a women loyalist who used letter - diaries to spy and transfer information about Rebels in PA to loyalist in New York James Rivington 1734 - 1802
Loyalist editor of Rivington Gazette and a Patriot Spy Colonel Beverley Robinson ( 1723 -- 1792 ) , of New York , Loyal American Regiment Lieutenant - Colonel
Robert Rogers ( 1731 -- 1795 ) , commander of Rogers ' Rangers / Queen 's Rangers to 1777 ( now The Queen 's York Rangers ( 1st American Regiment ) ( RCAC ) ) ,
innovator of ranging tactics Peggy Shippen ( 1760 -- 1804 ) , Philadelphia socialite and second wife of Benedict Arnold John Graves Simcoe ( 1752 -- 1806 ) ,
commander of Queen 's Rangers from 1777 ( now The Queen 's York Rangers ( 1st American Regiment ) ( RCAC ) ,
</item 1>

<item 2>
Loyalist (American Revolution)
Loyalist ( American Revolution ) - wikipedia Loyalist ( American Revolution ) Jump to : navigation , search Part of the Politics series on Toryism
Characteristics ( show ) Monarchism Traditionalism High Church Anglicanism Traditional Catholicism Loyalism Royalism Interventionism Agrarianism / Self -
sufficiency Counterrevolution Classicism High culture Noblesse oblige Organic unity Unionism General topics ( show ) Cavaliers Cavalier Parliament Château
Clique Jacobitism Divine right of kings Corporatism Family Compact ( Canada ) Oxford Movement Powellism People ( show ) Robert Filmer 1st Earl of Clarendon
Roger L'Estrange 1st Earl of Rochester 1st Viscount Bolingbroke 3rd Earl of Bute 1st Duke of Wellington Walter Scott Stanley Baldwin G.K.
</item 2>

<item 3>
Loyalist (American Revolution)
Benjamin Franklin to Baron Francis Maseres , June 26 , 1785 ( Opinion of Benjamin Franklin on persons who called themselves `` Loyalists '' , whom he judged
better called `` Royalists '' ) Bibliography of the Loyalist Participation in the American Revolution compiled by the United States Army Center of Military
History `` Black Loyalists : Our History , Our People '' Haldimand Collection The main source for historians in the study of the settlement of the American
Loyalists in Canada . More than 20,000 letters and documents , now fully indexed , and free on the Web . James Chalmers and `` Plain Truth '' ( A Loyalist
Answers Thomas Paine ) The Loyalist Declaration of Independence published in The Royal Gazette , ( New York ) on November 17 , 1781 ( Transcription provided by
Bruce Wallace and posted on The On - Line Institute for Advanced Loyalist Studies . )
</item 3>

<item 4>
Loyalist (American Revolution)
Martin v. Hunter 's Lessee Maryland Loyalists Battalion Tory Treaty of Paris ( 1783 ) United Empire Loyalist References ( edit ) Jump up ^ Barbara Smith ( 2013
) . The Freedoms We Lost : Consent and Resistance in Revolutionary America . New Press . p. 142 . Jump up ^ Jack P. Greene and J.R. Pole , eds , A Companion to
the American Revolution ( 2004 ) pp. 246 , 399 , 641 -- 2 Jump up ^ Calhoon , `` Loyalism and neutrality '' , p. 235 ; Middlekauff ( 2005 ) pp. 563 -- 564 ;
Thomas B. Allen , Tories : Fighting for the King in America 's First Civil War ( 2010 ) p. xx Jump up ^ Andrews , p. 284 Jump up ^ Jassanoff , ch 1 Jump up ^
Robinson , Charles Walker . Life of Sir John Beverly Robinson , bart. , C.B. , D.C.L. , chief - justice of Upper Canada . 1904 , page 469 .
</item 4>

<item 5>
Loyalist (American Revolution)
8 , No. 1 ( October 2007 ) . Why Were Some of Our Ancestors Tories ? Videos ( edit ) a lecture by Maya Jasanoff . `` An Imperial Disaster ? The Loyalist
Diaspora after the American Revolution '' . youtube . Thomas B. Allen . `` Video discussing his book '' . Tories : Fighting for the King in America 's First
Civil War . Thomas B. Allen . `` Lecture on his book at the Library of Congress '' . Tories : Fighting for the King in America 's First Civil War . YaleCourses
YouTube ( March 18 , 2011 ) . `` Who Were the Loyalists ? ( Youtube video ) '' . The American Revolution ( HIST 116 ) , Session 9 ( of 25 ) . Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Loyalist_(American_Revolution)&oldid=838344662 '' Categories : Loyalists in the American Revolution African Americans
in the American Revolution American diaspora American exiles American people of the American Revolution American refugees American Revolution American
Revolutionary War Black
</item 5>

<item 6>
Loyalist (American Revolution)
Chesterton Winston Churchill Enoch Powell George Grant Related topics ( show ) Conservatism Distributists Miguelism Vendéens Chouans Legitimism Carlism
Sanfedismo Viva Maria Cristeros Reactionary High Tory Red Tory Loyalist Royalist Veronese Easter Tory corporatism Tory socialism Pink Tory Ultra-Tories
Britannia offers solace and a promise of compensation for her exiled American - born Loyalists . ( Reception of the American Loyalists by Great Britain in the
Year 1783 . Engraving by H. Moses after Benjamin West . ) Loyalists were American colonists who remained loyal to the British Crown during the American
Revolutionary War . At the time they were often called Tories , Royalists , or King 's Men . They were opposed by the Patriots , those who supported the
revolution and called them `` persons inimical to the liberties of America '' . Prominent Loyalists repeatedly assured the British government that many
thousands of loyalists would spring to arms and fight for the crown . The British government acted in expectation of that , especially in the southern campaigns
in 1780 - 81 .
</item 6>

<item 7>
Loyalist (American Revolution)
Many active Church of England members became Loyalists . Some recent arrivals from Britain , especially those from Scotland , had a high Loyalist proportion .
Loyalists in the southern colonies were suppressed by the local Patriots , who controlled local and state government . Many people -- including former
Regulators in North Carolina -- refused to join the rebellion , as they had earlier protested against corruption by local authorities who later became
Revolutionary leaders . The oppression by the local Whigs during the Regulation led to many of the residents of backcountry North Carolina sitting out the
Revolution or siding with the Loyalists . In areas under Patriot control , Loyalists were subject to confiscation of property , and outspoken supporters of the
king were threatened with public humiliation such as tarring and feathering , or physical attack . It is not known how many Loyalist civilians were harassed by
the Patriots , but the treatment was a warning to other Loyalists not to take up arms . In September 1775 , William Drayton and Loyalist leader Colonel Thomas
Fletchall signed a treaty of neutrality in the interior community of Ninety Six , South Carolina .
</item 7>

<item 8>
Loyalist (American Revolution)
They felt themselves to be weak or threatened within American society and in need of an outside defender such as the British Crown and Parliament . They had
been promised freedom from slavery by the British . They felt that being a part of the British Empire was crucial in terms of commerce and their business
operations . Loyalism and military operations ( edit ) See also : Loyalists fighting in the American Revolution In the opening months of the Revolutionary War ,
the Patriots laid siege to Boston , where most of the British forces were stationed . Elsewhere there were few British troops and the Patriots seized control of
all levels of government , as well as supplies of arms and gunpowder . These actions were not without resistance . Especially in New York , New Jersey , and
parts of North and South Carolina , there was considerable ambivalence about the Patriot cause . Vocal Loyalists , often with the encouragement and assistance
of royal governors , recruited people to their side . In the South Carolina backcountry Loyalist recruitment oustripped that of Patriots .
</item 8>

<item 9>
Loyalist (American Revolution)
Many , like Maryland lawyer Daniel Dulaney , opposed taxation without representation , but would not break their oath to the King or take up arms against him .
In one of his many pamphlets , Dulaney wrote : `` There may be a time when redress may not be obtained . Till then , I shall recommend a legal , orderly , and
prudent resentment '' . Most hoped for a peaceful reconciliation , and were forced to choose sides by the Patriots who took control nearly everywhere in 1775 -
76 . Likely the earliest formal meeting , and use of the term `` royalist '' and the source of the United Empire Loyalist ( `` UEL '' ) acronym , took place in
Boston on October 28 , 1775 . At the meeting , held at the suggestion of General Thomas Gage , these Loyalists formed a society called `` The Loyalist
Associators despite the Unity of the Empire '' .
</item 9>

<item 10>
Loyalist (American Revolution)
They were alienated when the Patriots resorted to violence , such as burning houses and tarring and feathering . They wanted to take a middle - of - the road
position and were angry when forced by the Patriots to declare their opposition . They had a long - standing sentimental attachment to Britain ( often with
business and family links ) . They were procrastinators who realized that independence was bound to come someday , but wanted to postpone the moment . They were
cautious and afraid that chaos and mob rule would result . Some were pessimists who lacked the confidence in the future displayed by the Patriots . Others
recalled the dreadful experiences of many Jacobite rebels after the failure of the last Jacobite rebellion as recently as 1745 who often lost their lands when
the Hanoverian government won . Other motives of the Loyalists included : They felt a need for order and believed that Parliament was the legitimate authority .
In New York , powerful families had assembled colony - wide coalitions of supporters , Men long associated with the De Lancey faction went along when its
leadership decided to support the crown .
</item 10>
=== END ITEM id=427551a8f2170b45 ===

=== ITEM id=64579abda20e025a ===
Gold sentences and facts:
Sentence 0: Captain : Christopher Jones . About age 50 , of Harwich , a seaport in Essex , England , which was also the port of his ship Mayflower . He and his ship were veterans of the European cargo business , often carrying wine to England , but neither had ever crossed the Atlantic . By June 1620 , he and the Mayflower had been hired for the Pilgrims voyage by their business agents in London , Thomas Weston of the Merchant Adventurers and Robert Cushman . Historical marker in London honoring the Mayflower and Captain Jones .
  Fact 0.0: Christopher Jones was the captain of the Mayflower.
  Fact 0.1: Christopher Jones was about age 50.
  Fact 0.2: Christopher Jones was from Harwich, a seaport in Essex, England.
  Fact 0.3: Harwich was also the port of Christopher Jones's ship, the Mayflower.
  Fact 0.4: Christopher Jones and the Mayflower were veterans of the European cargo business.
  Fact 0.5: Christopher Jones and the Mayflower often carried wine to England.
  Fact 0.6: Neither Christopher Jones nor the Mayflower had ever crossed the Atlantic.
  Fact 0.7: By June 1620, Christopher Jones and the Mayflower had been hired for the Pilgrims' voyage by their business agents in London.
  Fact 0.8: Thomas Weston of the Merchant Adventurers was one of the business agents in London who hired the Mayflower for the Pilgrims' voyage.
  Fact 0.9: Robert Cushman was one of the business agents in London who hired the Mayflower for the Pilgrims' voyage.
  Fact 0.10: There is a historical marker in London honoring the Mayflower.
  Fact 0.11: There is a historical marker in London honoring Captain Jones.
Sentence 1: Captain Jones became master of the Mayflower 11 years prior to the Pilgrims ' voyage , sailing the ship cross-Channel taking English woolens to France and bringing French wine to London . In addition , he had also transported hats , hemp , Spanish salt , hops , and vinegar to Norway , and may have taken the Mayflower whaling in the North Atlantic in the Greenland area . She had traveled to Mediterranean ports , being then owned by Christopher Nichols , Robert Child , Thomas Short , and Christopher Jones , the ship 's master . In 1620 , Jones and Robert Child still owned their quarter shares in the ship , and it was from them that Thomas Weston chartered her in the summer of 1620 to undertake the Pilgrim voyage . Weston had a significant role in the Mayflower voyage , due to his membership in the Company of Merchant Adventurers , and he eventually traveled to the Plymouth Colony himself .
  Fact 1.0: Captain Jones became master of the Mayflower 11 years before the Pilgrims' voyage.
  Fact 1.1: Captain Jones sailed the Mayflower cross-Channel taking English woolens to France.
  Fact 1.2: Captain Jones sailed the Mayflower bringing French wine to London.
  Fact 1.3: Captain Jones also transported hats, hemp, Spanish salt, hops, and vinegar to Norway.
  Fact 1.4: Captain Jones may have taken the Mayflower whaling in the North Atlantic in the Greenland area.
  Fact 1.5: The Mayflower had traveled to Mediterranean ports.
  Fact 1.6: The Mayflower was owned by Christopher Nichols, Robert Child, Thomas Short, and Christopher Jones.
  Fact 1.7: Christopher Jones was the ship's master.
  Fact 1.8: In 1620, Christopher Jones still owned a quarter share in the Mayflower.
  Fact 1.9: In 1620, Robert Child still owned a quarter share in the Mayflower.
  Fact 1.10: Thomas Weston chartered the Mayflower from Jones and Child in the summer of 1620 to undertake the Pilgrim voyage.
  Fact 1.11: Weston had a significant role in the Mayflower voyage due to his membership in the Company of Merchant Adventurers.
  Fact 1.12: Thomas Weston eventually traveled to the Plymouth Colony himself.
Sentence 2: Aft on the main deck in the stern was the cabin for Master Christopher Jones , measuring about ten by seven feet ( 3 m × 2.1 m ) . Forward of that was the steerage room , which housed a whipstaff ( tiller extension ) for sailing control ; not a wheel , as in later ships . Also here was the ship 's compass and probably also berths for the ship 's officers . Forward of the steerage room was the capstan , a vertical axle used to pull in ropes or cables . Far forward on the main deck , just aft of the bow , was the forecastle space where the ship 's cook prepared meals for the crew ; it may also have been where the ship 's sailors slept .
  Fact 2.0: The cabin for Master Christopher Jones was aft on the main deck in the stern.
  Fact 2.1: The cabin for Master Christopher Jones measured about ten by seven feet (3 m × 2.1 m).
  Fact 2.2: The steerage room was forward of the cabin.
  Fact 2.3: The steerage room housed a whipstaff (tiller extension) for sailing control.
  Fact 2.4: The whipstaff was not a wheel, unlike in later ships.
  Fact 2.5: The ship's compass was also in the same area as the steerage room.
  Fact 2.6: The ship probably also had berths for the ship's officers in the steerage room area.
  Fact 2.7: The capstan was forward of the steerage room.
  Fact 2.8: The capstan was a vertical axle used to pull in ropes or cables.
  Fact 2.9: The forecastle space was far forward on the main deck, just aft of the bow.
  Fact 2.10: The ship's cook prepared meals for the crew in the forecastle space.
  Fact 2.11: The ship's sailors may also have slept in the forecastle space.

Retrieved items:
<item 1>
Mayflower
In addition , he had also transported hats , hemp , Spanish salt , hops , and vinegar to Norway , and may have taken the Mayflower whaling in the North Atlantic
in the Greenland area . She had traveled to Mediterranean ports , being then owned by Christopher Nichols , Robert Child , Thomas Short , and Christopher Jones
, the ship 's master . In 1620 , Jones and Robert Child still owned their quarter shares in the ship , and it was from them that Thomas Weston chartered her in
the summer of 1620 to undertake the Pilgrim voyage . Weston had a significant role in the Mayflower voyage , due to his membership in the Company of Merchant
Adventurers , and he eventually traveled to the Plymouth Colony himself . There were 26 vessels bearing the same name as the Pilgrim ship in the Port Books of
England in the reign of James I ( 1603 -- 1625 ) , and the reason for this popularity of the name has never been found . One particular Mayflower that has
caused historical confusion is a Mayflower erroneously identified as one of the 1620 Pilgrims .
</item 1>

<item 2>
Graph query row:
person: Christopher Jones
connected_entity: Mayflower
relationship_type: RELATED_TO
role: part owner
context: master of the Mayflower
relationship: None
relation: None
fact: None
</item 2>

<item 3>
Mayflower
He was named in Mourt 's Relation , London 1622 , under a date of January 12 , 1621 as a leader of an expedition to rescue Pilgrims lost in the forest for
several days while searching for housing - roof thatch . It is unknown in what capacity he came to the Mayflower and his given name is unknown . The title of ``
Master '' indicates that he was a person of some authority and prominence in the company . He may have been a principal officer of the Mayflower . No more is
known of him ; he may have returned to England on the Mayflower 's April 1621 voyage or died of the illnesses that affected so many that first winter . Later
history Three of Mayflower 's owners applied to the Admiralty court for an appraisal of the ship on May 4 , 1624 , two years after Captain Jones ' death in 1622
; one of these applicants was Jones ' widow Mrs. Josian ( Joan ) Jones . This appraisal probably was made to determine the valuation of the ship for the purpose
of settling the estate of its late master .
</item 3>

<item 4>
Mayflower
Mayflower - wikipedia Mayflower Jump to : navigation , search For other ships , see Mayflower ( ship ) . For other uses , see Mayflower ( disambiguation ) .
Mayflower in Plymouth Harbor by William Halsall ( 1882 ) Name : Mayflower Owner : Christopher Jones ( 1⁄4 of the ship ) Maiden voyage : Before 1609 Out of
service : 1622 -- 1624 Fate : most likely taken apart by Rotherhithe shipbreaker c. 1624 . General characteristics Class and type : Dutch cargo fluyt Tonnage :
180 tons + Length : c. 80 -- 90 ft ( 24 -- 27.5 m ) on deck , 100 -- 110 ft ( 30 -- 33.5 m ) overall . Decks : Around 4 Capacity : Unknown , but carried c. 135
people during the historical voyage to what they would call Plymouth Colony The Mayflower was an English ship that famously transported the first English
Puritans , known today as the Pilgrims , from Plymouth , England to the New World in 1620 .
</item 4>

<item 5>
Mayflower
Time was measured with the ancient method of an hourglass . Arrival in America On November 9 , 1620 , they sighted present - day Cape Cod . They spent several
days trying to sail south to their planned destination of the Colony of Virginia , where they had obtained permission to settle from the Company of Merchant
Adventurers . However , strong winter seas forced them to return to the harbor at Cape Cod hook , well north of the intended area , where they anchored on
November 11 . The settlers wrote and signed the Mayflower Compact after the ship dropped anchor at Cape Cod , in what is now Provincetown Harbor , in order to
establish legal order and to quell increasing strife within the ranks . On Monday , November 27 , an exploring expedition was launched under the direction of
Capt . Christopher Jones to search for a suitable settlement site . As master of the Mayflower , Jones was not required to assist in the search , but he
apparently thought it in his best interest to assist the search expedition . There were 34 persons in the open shallop : 24 passengers and 10 sailors .
</item 5>

<item 6>
Mayflower
Nathaniel Philbrick estimates between 20 and 30 sailors in her crew whose names are unknown . Nick Bunker states that Mayflower had a crew of at least 17 and
possibly as many as 30 . Caleb Johnson states that the ship carried a crew of about 30 men , but the exact number is unknown . Officers and crew Captain :
Christopher Jones . About age 50 , of Harwich , a seaport in Essex , England , which was also the port of his ship Mayflower . He and his ship were veterans of
the European cargo business , often carrying wine to England , but neither had ever crossed the Atlantic . By June 1620 , he and the Mayflower had been hired
for the Pilgrims voyage by their business agents in London , Thomas Weston of the Merchant Adventurers and Robert Cushman . Historical marker in London honoring
the Mayflower and Captain Jones . Masters Mate : John Clark ( Clarke ) , Pilot . By age 45 in 1620 , Clark already had greater adventures than most other
mariners of that dangerous era . His piloting career began in England about 1609 .
</item 6>

<item 7>
Mayflower
In 1624 , Thomas Russell supposedly added to part of a farmhouse already there with timbers from a ship , believed to be from the Pilgrim ship Mayflower ,
bought from a shipbreaker 's yard in Rotherhithe . The well - preserved structure was a tourist attraction , receiving visitors each year from all over the
world and particularly from America , but it is now privately owned and not open to the public . Second Mayflower Another ship called the Mayflower made a
voyage from London to Plymouth Colony in 1629 carrying 35 passengers , many from the Pilgrim congregation in Leiden that organized the first voyage . This was
not the same ship that made the original voyage with the first settlers . This voyage began in May and reached Plymouth in August . This ship also made the
crossing from England to America in 1630 ( as part of the Winthrop Fleet ) , 1633 , 1634 , and 1639 . It attempted the trip again in 1641 , departing London in
October of that year under master John Cole , with 140 passengers bound for Virginia . It never arrived . On October 18 , 1642 , a deposition was made in
England regarding the loss .
</item 7>

<item 8>
Mayflower
Other items included most of their clothing and bedding . The hold also stored the passengers ' personal weapons and military equipment -- armor , muskets ,
gunpowder , and shot , as well as swords and bandoliers . It also stored all the tools that the Pilgrims would need , as well as all the equipment and utensils
needed to prepare meals in the New World . It is also known that some Pilgrims loaded trade goods on board , including Isaac Allerton , William Mullins , and
possibly others ; these also most likely were stored in the cargo hold . Early history It is not known when and where the Mayflower was built , but it is likely
that she was launched at Harwich in the county of Essex , England . She was designated as `` of Harwich '' in the Port Books of 1609 -- 11 , although later
known as `` of London '' . Harwich was also the birthplace of Mayflower master Christopher Jones about 1570 . Captain Jones became master of the Mayflower 11
years prior to the Pilgrims ' voyage , sailing the ship cross-Channel taking English woolens to France and bringing French wine to London .
</item 8>

<item 9>
Mayflower
Or some of the records of the period may have been lost . Voyage Speedwell and Mayflower Mayflower arrived in the `` Cape Cod fishhook '' , 11 November 1620 (
satellite photo , 1997 ) Approximately 65 passengers embarked the Mayflower in the middle of July 1620 at its home port in the Rotherhithe district of London on
the Thames river . The ship then proceeded down the Thames into the English Channel and then on to the south coast of England to anchor at Southampton Water .
The Mayflower waited there for a rendezvous on July 22 with the Speedwell . Speedwell was coming from Holland with English separatist Puritans , members of the
Leiden congregation who had been living in Holland to escape religious persecution in England . The two ships set sail for America around August 5 , but the
Speedwell sprang a leak shortly after , and the two ships were brought into Dartmouth for repairs . They made a new start after the repairs , and they were more
than 200 miles ( 320 km ) beyond Land 's End at the southwestern tip of England when Speedwell sprang another leak .
</item 9>

<item 10>
Mayflower
At his death , his estate consisted of extensive footwear and other items of clothing , and made his daughter Priscilla and her husband John Alden quite
prosperous . No cattle or beasts of draft or burden were brought on the journey , but there were pigs , goats , and poultry . Some passengers brought family
pets such as cats and birds . Peter Browne took his large bitch mastiff , and John Goodman brought along his spaniel . Mayflower officers , crew , and others
According to author Charles Banks , the officers and crew of the Mayflower consisted of a captain , four mates , four quartermasters , surgeon , carpenter ,
cooper , cooks , boatswains , gunners , and about 36 men before the mast , making a total of about 50 . The entire crew stayed with the Mayflower in Plymouth
through the winter of 1620 -- 1621 , and about half of them died during that time . The remaining crewmen returned to England on the Mayflower , which sailed
for London on April 5 , 1621 . Crew members per various sources Banks states that the crew totaled 36 men before the mast and 14 officers , making a total of 50
.
</item 10>
=== END ITEM id=64579abda20e025a ===

=== ITEM id=945b82b55c4f1517 ===
Gold sentences and facts:
Sentence 0: The gymnosperms are a group of seed - producing plants that includes conifers , cycads , Ginkgo , and gnetophytes . The term `` gymnosperm '' comes from the Greek composite word γυμνόσπερμος ( γυμνός gymnos , `` naked '' and σπέρμα sperma , `` seed '' ) , meaning `` naked seeds '' . The name is based on the unenclosed condition of their seeds ( called ovules in their unfertilized state ) . The non-encased condition of their seeds stands in contrast to the seeds and ovules of flowering plants ( angiosperms ) , which are enclosed within an ovary . Gymnosperm seeds develop either on the surface of scales or leaves , which are often modified to form cones , or solitary as in Yew , Torreya , Ginkgo .
  Fact 0.0: The gymnosperms are a group of seed-producing plants.
  Fact 0.1: Gymnosperms include conifers, cycads, Ginkgo, and gnetophytes.
  Fact 0.2: The term gymnosperm comes from the Greek composite word γυμνόσπερμος.
  Fact 0.3: The Greek word γυμνός (gymnos) means naked.
  Fact 0.4: The Greek word σπέρμα (sperma) means seed.
  Fact 0.5: The term gymnosperm means naked seeds.
  Fact 0.6: The name gymnosperm is based on the unenclosed condition of gymnosperm seeds.
  Fact 0.7: Gymnosperm ovules are called ovules in their unfertilized state.
  Fact 0.8: Gymnosperm seeds are not encased.
  Fact 0.9: The seeds and ovules of flowering plants (angiosperms) are enclosed within an ovary.
  Fact 0.10: Gymnosperm seeds develop either on the surface of scales or leaves.
  Fact 0.11: The scales or leaves on which gymnosperm seeds develop are often modified to form cones.
  Fact 0.12: Gymnosperm seeds can also be solitary, as in Yew, Torreya, and Ginkgo.

Retrieved items:
<item 1>
Gymnosperm
After syngamy ( joining of the sperm and egg cell ) , the zygote develops into an embryo ( young sporophyte ) . More than one embryo is usually initiated in
each gymnosperm seed . The mature seed comprises the embryo and the remains of the female gametophyte , which serves as a food supply , and the seed coat .
Genetics ( edit ) The first published sequenced genome for any gymnosperm was the genome of Picea abies in 2013 . References ( edit ) ^ Jump up to : ``
Gymnosperms on The Plant List '' . Theplantlist.org . Retrieved 2013 - 07 - 24 . ^ Jump up to : Raven , P.H. ( 2013 ) . Biology of Plants . New York : W.H.
Freeman and Co . Jump up ^ Jeffrey D. Palmer ; Douglas E. Soltis ; Mark W. Chase ( 2004 ) . `` The plant tree of life : an overview and some points of view '' .
American Journal of Botany. 91 ( 10 ) : 1437 -- 1445 .
</item 1>

<item 2>
Gymnosperm
Two spore types , microspores and megaspores , are typically produced in pollen cones or ovulate cones , respectively . Gametophytes , as with all heterosporous
plants , develop within the spore wall . Pollen grains ( microgametophytes ) mature from microspores , and ultimately produce sperm cells . Megagametophytes
develop from megaspores and are retained within the ovule . Gymnosperms produce multiple archegonia , which produce the female gamete . During pollination ,
pollen grains are physically transferred between plants from the pollen cone to the ovule . Pollen is usually moved by wind or insects . Whole grains enter each
ovule through a microscopic gap in the ovule coat ( integument ) called the micropyle . The pollen grains mature further inside the ovule and produce sperm
cells . Two main modes of fertilization are found in gymnosperms . Cycads and Ginkgo have motile sperm that swim directly to the egg inside the ovule , whereas
conifers and gnetophytes have sperm with no flagella that are moved along a pollen tube to the egg .
</item 2>

<item 3>
Gymnosperm
External links ( edit ) Wikimedia Commons has media related to Gymnosperm . Gymnosperm Database Gymnosperms on the Tree of Life Albert Seward ( 1911 ) . ``
Gymnosperms '' . Encyclopædia Britannica ( 11th ed . ) . Botany History of botany Subdisciplines Plant systematics Ethnobotany Paleobotany Plant anatomy Plant
ecology Phytogeography Geobotany Flora Phytochemistry Plant pathology Bryology Phycology Floristics Dendrology Plant groups Algae Archaeplastida Bryophyte
Non-vascular plants Vascular plants Spermatophytes Pteridophyte Gymnosperm Angiosperm Plant morphology ( glossary ) Plant cells Cell wall Phragmoplast Plastid
Plasmodesma Vacuole Tissues Meristem Vascular tissue Vascular bundle Ground tissue Mesophyll Cork Wood Storage organs Vegetative Root Rhizoid Bulb Rhizome Shoot
Stem Leaf Petiole Cataphyll Bud Sessility Reproductive ( Flower ) Flower development Inflorescence Umbel Raceme Bract Pedicellate Flower Whorl Floral symmetry
Floral diagram Floral formula Receptacle
</item 3>

<item 4>
Gymnosperm
Gymnosperm - wikipedia Gymnosperm Jump to : navigation , search Gymnosperms Temporal range : 370 -- 0 Ma PreЄ Є Pg Devonian - Present Various gymnosperms .
Scientific classification Kingdom : Plantae Subkingdom : Embryophyta ( unranked ) : Gymnospermae ( paraphyletic ) Divisions Pinophyta ( or Coniferophyta ) -
Conifers Ginkgophyta - Ginkgo Cycadophyta - Cycads Gnetophyta - Gnetum , Ephedra , Welwitschia Encephalartos sclavoi cone , about 30 cm long The gymnosperms are
a group of seed - producing plants that includes conifers , cycads , Ginkgo , and gnetophytes . The term `` gymnosperm '' comes from the Greek composite word
γυμνόσπερμος ( γυμνός gymnos , `` naked '' and σπέρμα sperma , `` seed '' ) , meaning `` naked seeds '' .
</item 4>

<item 5>
Gymnosperm
Agathis in Araucariaceae and Nageia in Podocarpaceae have broad , flat strap - shaped leaves . Cycads are the next most abundant group of gymnosperms , with two
or three families , 11 genera , and approximately 338 species . A majority of cycads are native to tropical climates and are most abundantly found in regions
near the equator . The other extant groups are the 95 - 100 species of Gnetales and one species of Ginkgo . Uses ( edit ) Gymnosperms have major economic uses .
Pine , fir , spruce , and cedar are all examples of conifers that are used for lumber , paper production , and resin . Some other common uses for gymnosperms
are soap , varnish , nail polish , food , gum , and perfumes . Life cycle ( edit ) Example of gymnosperm lifecycle Gymnosperms , like all vascular plants , have
a sporophyte - dominant life cycle , which means they spend most of their life cycle with diploid cells , while the gametophyte ( gamete - bearing phase ) is
relatively short - lived .
</item 5>

<item 6>
Gymnosperm
Diselma , Fitzroya , Callitris ( incl . Actinostrobus and Neocallitropsis ) , Thujopsis , Thuja , Fokienia , Chamaecyparis , Callitropsis , Cupressus ,
Juniperus , Xanthocyparis , Calocedrus , Tetraclinis , Platycladus , Microbiota Family Taxaceae : Austrotaxus , Pseudotaxus , Taxus , Cephalotaxus , Amentotaxus
, Torreya Diversity and origin ( edit ) There are more than 1000 extant or currently living species of gymnosperms in 88 plant genera belonging to 14 plant
families . It is widely accepted that the gymnosperms originated in the late Carboniferous period , replacing the lycopsid rainforests of the tropical region .
This appears to have been the result of a whole genome duplication event around 319 million years ago . Early characteristics of seed plants were evident in
fossil progymnosperms of the late Devonian period around 383 million years ago .
</item 6>

<item 7>
Gymnosperm
There is conflicting evidence on the question of whether the living gymnosperms form a clade . The fossil record of gymnosperms includes many distinctive taxa
that do not belong to the four modern groups , including seed - bearing trees that have a somewhat fern - like vegetative morphology ( the so - called `` seed
ferns '' or pteridosperms . ) When fossil gymnosperms such as Bennettitales , Caytonia and the glossopterids are considered , it is clear that angiosperms are
nested within a larger gymnosperm clade , although which group of gymnosperms is their closest relative remains unclear . For the most recent classification on
extant gymnosperms see Christenhusz et al. ( 2011 ) . There are 12 families , 83 known genera with a total of ca 1080 known species ( Christenhusz & Byng 2016 )
.
</item 7>

<item 8>
Gymnosperm
91 ( 10 ) : 1437 -- 1445 . doi : 10.3732 / ajb. 91.10. 1437 . PMID 21652302 . Jump up ^ Stevens , P.F. `` Angiosperm Phylogeny Website - Seed Plant Evolution ''
. Jump up ^ Hilton , Jason , and Richard M. Bateman. 2006 . Pteridosperms are the backbone of seed - plant phylogeny . Journal of the Torrey Botanical Society
133 : 119 - 168 ( abstract ) Jump up ^ Christenhusz , M.J.M. ; Reveal , J.L. ; Farjon , A. ; Gardner , M.F. ; Mill , R.R. ; Chase , M.W. ( 2011 ) . `` A new
classification and linear sequence of extant gymnosperms '' ( PDF ) . Phytotaxa. 19 : 55 -- 70 . Jump up ^ Christenhusz , M.J.M. ; Byng , J.W. ( 2016 ) .
</item 8>

<item 9>
Gymnosperm
The name is based on the unenclosed condition of their seeds ( called ovules in their unfertilized state ) . The non-encased condition of their seeds stands in
contrast to the seeds and ovules of flowering plants ( angiosperms ) , which are enclosed within an ovary . Gymnosperm seeds develop either on the surface of
scales or leaves , which are often modified to form cones , or solitary as in Yew , Torreya , Ginkgo . The gymnosperms and angiosperms together compose the
spermatophytes or seed plants . The gymnosperms are divided into six phyla . Organisms that belong to the Cycadophyta , Ginkgophyta , Gnetophyta , and Pinophyta
( also known as Coniferophyta ) phyla are still in existence while those in the Pteridospermales and Cordaitales phyla are now extinct .
</item 9>

<item 10>
Gymnosperm
By far the largest group of living gymnosperms are the conifers ( pines , cypresses , and relatives ) , followed by cycads , gnetophytes ( Gnetum , Ephedra and
Welwitschia ) , and Ginkgo biloba ( a single living species ) . Roots in some genera have fungal association with roots in the form of mycorrhiza ( Pinus ) ,
while in some others ( Cycas ) small specialised roots called coralloid roots are associated with nitrogen - fixing cyanobacteria . Contents ( hide ) 1
Classification 2 Diversity and origin 3 Uses 4 Life cycle 5 Genetics 6 References 7 External links Classification ( edit ) Further information : Spermatophyte
In early classification schemes , the gymnosperms ( Gymnospermae ) were regarded as a `` natural '' group . There is conflicting evidence on the question of
whether the living gymnosperms form a clade .
</item 10>
=== END ITEM id=945b82b55c4f1517 ===

=== ITEM id=00860276fe374cfc ===
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
China 's heavy artillery facing Taiwan could then be moved to Tibet . It took China six to eight months to gather the resources needed for the war , according
to Anil Athale , author of the official Indian history . The Chinese sent a large quantity of non-military supplies to Tibet through the Indian port of Calcutta
. Early incidents ( edit ) Various border conflicts and `` military incidents '' between India and China flared up throughout the summer and autumn of 1962 . In
May , the Indian Air Force was told not to plan for close air support , although it was assessed as being a feasible way to counter the unfavourable ratio of
Chinese to Indian troops . In June , a skirmish caused the deaths of dozens of Chinese troops . The Indian Intelligence Bureau received information about a
Chinese buildup along the border which could be a precursor to war . During June -- July 1962 , Indian military planners began advocating `` probing actions ''
against the Chinese , and accordingly , moved mountain troops forward to cut off Chinese supply lines . According to Patterson , the Indian motives were
threefold : Test Chinese resolve and intentions regarding India .
</item 1>

<item 2>
Sino-Indian War
Xu Yan , prominent Chinese military historian and professor at the PLA 's National Defense University , gives an account of the Chinese leadership 's decision
to go to war . By late September 1962 , the Chinese leadership had begun to reconsider their policy of `` armed coexistence '' , which had failed to address
their concerns with the forward policy and Tibet , and consider a large , decisive strike . On 22 September 1962 , the People 's Daily published an article
which claimed that `` the Chinese people were burning with ' great indignation ' over the Indian actions on the border and that New Delhi could not ' now say
that warning was not served in advance ' . '' Military planning ( edit ) The Indian side was confident war would not be triggered and made little preparations .
India had only two divisions of troops in the region of the conflict . In August 1962 , Brigadier D.K. Palit claimed that a war with China in the near future
could be ruled out . Even in September 1962 , when Indian troops were ordered to `` expel the Chinese '' from Thag La , Maj . General J.S.
</item 2>

<item 3>
Sino-Indian War
The Sino - Indian War ( Hindi : भारत - चीन युद्ध Bhārat - Chīn Yuddh ) , also known as the Sino - Indian Border Conflict ( simplified Chinese : 中 印 边境 战争 ;
traditional Chinese : 中 印 邊境 戰爭 ; pinyin : Zhōng - Yìn Biānjìng Zhànzhēng ) , was a war between China and India that occurred in 1962 . A disputed Himalayan
border was the main pretext for war , but other issues played a role . There had been a series of violent border incidents after the 1959 Tibetan uprising ,
when India had granted asylum to the Dalai Lama . India initiated a Forward Policy in which it placed outposts along the border , including several north of the
McMahon Line , the eastern portion of a Line of Actual Control proclaimed by Chinese Premier Zhou Enlai in 1959 .
</item 3>

<item 4>
Sino-Indian War
According to Patterson , the Indian motives were threefold : Test Chinese resolve and intentions regarding India . Test whether India would enjoy Soviet backing
in the event of a Sino - Indian war . Create sympathy for India within the U.S. , with whom relations had deteriorated after the Indian annexation of Goa . On
10 July 1962 , 350 Chinese troops surrounded an Indian post in Chushul ( north of the McMahon Line ) but withdrew after a heated argument via loudspeaker . On
22 July , the Forward Policy was extended to allow Indian troops to push back Chinese troops already established in disputed territory . Whereas Indian troops
were previously ordered to fire only in self - defence , all post commanders were now given discretion to open fire upon Chinese forces if threatened . In
August , the Chinese military improved its combat readiness along the McMahon Line and began stockpiling ammunition , weapons and gasoline . Given his
foreknowledge of the coming Cuban Missile Crisis , Mao Zedong was able to persuade Nikita Khrushchev to reverse the Russian policy of backing India , at least
temporarily . In mid-October , the Communist organ Pravda encouraged peace between India and China .
</item 4>

<item 5>
Sino-Indian War
The Indian forces were undermanned , with only an understrength battalion to support them , while the Chinese troops had three regiments positioned on the north
side of the river . The Indians expected Chinese forces to cross via one of five bridges over the river and defended those crossings . However , the PLA
bypassed the defenders by crossing the shallow October river instead . They formed up into battalions on the Indian - held south side of the river under cover
of darkness , with each battalion assigned against a separate group of Rajputs . At 5 : 14 am , Chinese mortar fire began attacking the Indian positions .
Simultaneously , the Chinese cut the Indian telephone lines , preventing the defenders from making contact with their headquarters . At about 6 : 30 am , the
Chinese infantry launched a surprise attack from the rear and forced the Indians to leave their trenches . The Chinese overwhelmed the Indian troops in a series
of flanking manoeuvres south of the McMahon Line and prompted their withdrawal from Namka Chu . Fearful of continued losses , Indian troops retreated into
Bhutan . Chinese forces respected the border and did not pursue .
</item 5>

<item 6>
Sino-Indian War
Kaul had then realised that the attack would be desperate and the Indian government tried to stop an escalation into all - out war . Indian troops marching to
Thag La had suffered in the previously unexperienced conditions ; two Gurkha soldiers died of pulmonary edema . On 10 October , an Indian Punjabi patrol of 50
troops to Yumtso La were met by an emplaced Chinese position of some 1,000 soldiers . Indian troops were in no position for battle , as Yumtso La was 16,000
feet ( 4,900 m ) above sea level and Kaul did not plan on having artillery support for the troops . The Chinese troops opened fire on the Indians under their
belief that they were north of the McMahon Line . The Indians were surrounded by Chinese positions which used mortar fire . However , they managed to hold off
the first Chinese assault , inflicting heavy casualties . At this point , the Indian troops were in a position to push the Chinese back with mortar and machine
gun fire . However , Brigadier Dalvi opted not to fire , as it would mean decimating the Rajput who were still in the area of the Chinese regrouping .
</item 6>

<item 7>
Sino-Indian War
Following the 20 October attack , nothing was heard from Galwan . On 24 October , Indian forces fought hard to hold the Rezang La Ridge , in order to prevent a
nearby airstrip from falling to the Chinese . After realising the magnitude of the attack , the Indian Western Command withdrew many of the isolated outposts to
the south - east . Daulet Beg Oldi was also evacuated , but it was south of the Chinese claim line and was not approached by Chinese forces . Indian troops were
withdrawn in order to consolidate and regroup in the event that China probed south of their claim line . Lull in the fighting ( edit ) By 24 October , the PLA
had entered territory previously administered by India to give the PRC a diplomatically strong position over India . The majority of Chinese forces had advanced
sixteen kilometres south of the control line prior to the conflict . Four days of fighting were followed by a three - week lull . Zhou ordered the troops to
stop advancing as he attempted to negotiate with Nehru . The Indian forces had retreated into more heavily fortified positions around Se La and Bombdi La which
would be difficult to assault .
</item 7>

<item 8>
Sino-Indian War
It is noteworthy that the buildup and offensive from China occurred concurrently with the 13 - day Cuban Missile Crisis ( 16 -- 28 October 1962 ) that saw both
the United States and the Soviet Union confronting each other , and India did not receive assistance from either of these world powers until the Cuban Missile
Crisis was resolved . Contents ( hide ) 1 Location 2 Background 2.1 Aksai Chin 2.2 The McMahon Line 3 Events leading up to war 3.1 Tibet and the border dispute
3.2 1960 meetings to resolve the boundary question 3.3 The Forward Policy 3.4 Early incidents 4 Confrontation at Thag La 5 Chinese and Indian preparations 5.1
Motives 5.2 Military planning 6 Chinese offensive 6.1 Eastern theatre 6.2 Western theatre 7 Lull in the fighting 8 Continuation of war 8.1 Eastern theatre 8.2
Western theatre 9 Ceasefire 10 World opinion 11 Involvement of other nations 12 Aftermath 12.1 China 12.2 India 12.2.
</item 8>

<item 9>
Sino-Indian War
( March 2017 ) ( Learn how and when to remove this template message ) Sino - Indian War ( 1962 ) Remembering a War : The 1962 India - China Conflict --
Rediff.com . Neville Maxwell : Henderson Brooks Report 1962 Sino - Indian War , Hindustan Times Why India lost the 1962 border war ? -- Tejas Patel War in the
Himalayas : 1962 Indo - Sino Conflict ( includes official war history ) from History Division , Ministry of Defence , Government of India Critical Asian Studies
Article : Sino Indian War 1962 India , China to speed up border dispute talks : 2005 Xinhuanet The Rediff Special / Claude Arpi 1962 War and Its Implications fr
Sino - India Relations Historical maps of the Sino - Indian border ( in Chinese ) Conflict in Kashmir : Selected Internet Resources by the Library , University
of California , Berkeley , USA ; ( ( University of California , Berkeley Library Bibliographies and Web - Bibliographies list ) Frontier India India - China
Section China , India , and the fruits of Nehru 's folly by Venkatesan Vembu , Daily News & Analysis , 6 June
</item 9>

<item 10>
Sino-Indian War
Unable to reach political accommodation on disputed territory along the 3,225 - kilometre - long Himalayan border , the Chinese launched simultaneous offensives
in Ladakh and across the McMahon Line on 20 October 1962 , coinciding with the Cuban Missile Crisis . Chinese troops advanced over Indian forces in both
theatres , capturing Rezang la in Chushul in the western theatre , as well as Tawang in the eastern theatre . The war ended when China declared a ceasefire on
20 November 1962 , and simultaneously announced its withdrawal to its claimed ' line of actual control ' . The Sino - Indian War is notable for the harsh
mountain conditions under which much of the battle took place , entailing large - scale combat at altitudes of over 4,000 metres ( 14,000 feet ) . The Sino -
Indian War was also noted for the non-deployment of the navy or air force by either the Chinese or Indian side .
</item 10>
=== END ITEM id=00860276fe374cfc ===

=== ITEM id=c1579870cdbe7f15 ===
Gold sentences and facts:
Sentence 0: `` Ca n't Get You Out of My Head '' is a song recorded by Australian singer Kylie Minogue for her eighth studio album , titled Fever , which she released in 2001 . The song was released in Australia by Parlophone as the lead single from the album on 8 September 2001 . It was released on 17 September 2001 in the United Kingdom . In the United States , the single was released on 18 February 2002 . Jointly written , composed , and produced by Cathy Dennis and Rob Davis , `` Ca n't Get You Out of My Head '' is a midtempo dance - pop song which lyrically details its narrator 's obsession towards her lover . The song is famous for its `` la la la '' hook .
  Fact 0.0: 'Can't Get You Out of My Head' is a song.
  Fact 0.1: 'Can't Get You Out of My Head' is recorded by Australian singer Kylie Minogue.
  Fact 0.2: 'Can't Get You Out of My Head' was recorded for Kylie Minogue's eighth studio album.
  Fact 0.3: The eighth studio album is titled Fever.
  Fact 0.4: Kylie Minogue released Fever in 2001.
  Fact 0.5: 'Can't Get You Out of My Head' was released in Australia by Parlophone.
  Fact 0.6: 'Can't Get You Out of My Head' was released in Australia as the lead single from Fever.
  Fact 0.7: The Australian release of 'Can't Get You Out of My Head' was on 8 September 2001.
  Fact 0.8: 'Can't Get You Out of My Head' was released in the United Kingdom on 17 September 2001.
  Fact 0.9: 'Can't Get You Out of My Head' was released in the United States as a single on 18 February 2002.
  Fact 0.10: 'Can't Get You Out of My Head' was jointly written by Cathy Dennis and Rob Davis.
  Fact 0.11: 'Can't Get You Out of My Head' was jointly composed by Cathy Dennis and Rob Davis.
  Fact 0.12: 'Can't Get You Out of My Head' was jointly produced by Cathy Dennis and Rob Davis.
  Fact 0.13: 'Can't Get You Out of My Head' is a midtempo dance-pop song.
  Fact 0.14: 'Can't Get You Out of My Head' lyrically details its narrator's obsession towards her lover.
  Fact 0.15: The song is famous for its 'la la la' hook.

Retrieved items:
<item 1>
Can't Get You Out of My Head
The music , excluding the guitar part , was programmed using Korg Triton workstation via MIDI . Dennis later remarked : `` Even though Kylie was n't the first
artist to be offered the song , I do n't believe it was meant to go to anyone other than Kylie , and I do n't believe anyone else would have done the incredible
job she did with it , with the video , looking super-hot ! '' Composition ( edit ) `` Ca n't Get You Out of My Head '' A 21 - second sample of `` Ca n't Get You
Out of My Head '' containing the hook and part of the chorus of the song . The song is known for its `` la la la '' hook . Problems playing this file ? See
media help . `` Ca n't Get You Out of My Head '' is a `` robotic '' midtempo dance - pop song , with a tempo of 126 beats per minute .
</item 1>

<item 2>
Graph query row:
writer: Rob Davis
relationship: CREATED
work: Can't Get You Out of My Head
</item 2>

<item 3>
Can't Get You Out of My Head
Jason Thompson from PopMatters described Minogue 's vocals as a `` sexual come on '' and called the song `` trim and funky , certainly something that could n't
miss anywhere . '' Dominique Leone from Pitchfork Media praised the commercial prospect of the song , saying that it `` exudes a catchiness that belies its
inherent simplicity , so reassuring during an era when chart acts sound increasingly baroque and producers race to see who can ape electronic music trends first
. '' Jim Farber from Entertainment Weekly said that the song `` fully lives up to its title '' with `` every sound a hook , '' and compared it to the works of
Andrea True . Michael Hubbard from MusicOMH labelled the song `` one of 2001 's best singles , '' saying that it `` predictably beat off lesser ' competition .
' '' In 2012 , The Guardian critic Everett True defined `` Ca n't Get You Out of My Head '' as `` one of those rare moments in pop : sleek and chic and stylish
and damnably danceable , but with a darker element hidden in plain sight . ''
</item 3>

<item 4>
Can't Get You Out of My Head
In 2012 , Priya Elan from NME ranked the song at number four on his `` The Greatest Pop Songs in History '' list , saying : `` It was unlike any song I remember
hearing before . '' In 2012 , The Guardian included the song in their list of `` The Best Number One Records , '' labelling it to be `` sleek , Arctic - blue
minimalism , like an emotionally thwarted retelling of Donna Summer 's `` I Feel Love . '' In the same year , PRS for Music , a UK copyright collection society
and performance rights organisation which collects royalties on behalf of songwriters and composers , named `` Ca n't Get You Out of My Head '' the `` Most
Popular Song of the Decade '' as it received the highest airplay and live covers in the 2000s ( decade ) . Minogue performing `` Ca n't Get You Out of My Head
'' during her KylieX Tour ( 2008 ) .
</item 4>

<item 5>
Can't Get You Out of My Head
In 2013 , a survey of 700 people was conducted as part of the Manchester Science Festival to find out which song they considered the `` catchiest , '' and `` Ca
n't Get You Out of My Head '' topped the poll . Lee Barron , in his essay The Seven Ages of Kylie Minogue : Postmodernism , Identity , and Performative Mimicry
, noted that the song `` further established Minogue 's cultural and commercial relevance in the new millennium . '' He remarked that the song `` with its
hypnotic ' la la la ' refrain and the deceptively uncomplicated , catchily repetitive beats and synth - sound , marked yet another clearly defined image
transformation from the camp - infused Light Years to an emphasis upon a cool , machine - like sexuality , a trait clearly identifiable within the promotional
video for ' Ca n't Get You Out of My Head . '
</item 5>

<item 6>
Can't Get You Out of My Head
It was released on 17 September 2001 in the United Kingdom . In the United States , the single was released on 18 February 2002 . Jointly written , composed ,
and produced by Cathy Dennis and Rob Davis , `` Ca n't Get You Out of My Head '' is a midtempo dance - pop song which lyrically details its narrator 's
obsession towards her lover . The song is famous for its `` la la la '' hook . In addition to positive reviews from music critics , `` Ca n't Get You Out of My
Head '' found commercial success on a large scale . It peaked at number one on the charts of Austria , Belgium , France , Germany , Italy , Poland , Switzerland
, the United Kingdom , and every other European country excluding Finland . It also topped the charts of Minogue 's native country Australia , Canada and New
Zealand . In the United States , the song peaked at number seven on the US Billboard Hot 100 chart , becoming Minogue 's biggest hit in the region since `` The
Loco - Motion '' . `` Ca n't Get You Out of My Head '' reportedly reached number one in 40 countries across the globe .
</item 6>

<item 7>
Can't Get You Out of My Head
My A&R at the time , Miles Leonard and Jamie Nelson , said , `` We 've got something . Come into the office . We 've got to play you something . '' So I went to
the EMI office , and I had the same reaction you did , about 20 seconds in . I could n't even fathom what I was hearing . It just ... did something . I was
beside myself . Then at the end of the song , panic set in . I was saying `` Are you sure we 've got this song ? Do n't tell me that we do n't ! Is it secured ?
Can we have it ? '' And we did ! And that kick - started a whole different phase in my career . -- Minogue , about her reaction to hearing the demo of the song
. `` Ca n't Get You Out of My Head '' was jointly written , composed , and produced by Cathy Dennis and Rob Davis . Dennis and Davis had been brought together
by British artist manager Simon Fuller , who wanted the duo to come up with a song for British pop group S Club 7 .
</item 7>

<item 8>
Graph query row:
writer: Cathy Dennis
relationship: CREATED
work: Can't Get You Out of My Head
</item 8>

<item 9>
Can't Get You Out of My Head
Ca n't Get You Out of My Head - Wikipedia Ca n't Get You Out of My Head Jump to : navigation , search `` Ca n't Get You Out of My Head '' Single cover used for
CD # 02 and digital releases of the song .
</item 9>

<item 10>
Can't Get You Out of My Head
Through the lyrics of the song , its singer expresses an obsession with an anonymous figure . Dorian Lynskey from The Guardian termed the song a `` mystery ''
as Minogue never reveals the identity of her object of infatuation . The critic suggested that the person Minogue is referring to is either `` a partner , an
evasive one - night stand or someone who does n't know she exists . '' Writing for the same newspaper , Everett True identified a `` darker element '' in the
simple lyrics and felt this sentiment was echoed in Minogue 's restrained vocals . Further , True emphasised that while Minogue 's 1987 single `` I Should Be So
Lucky '' had presented an optimistic romantic future , `` Ca n't Get You Out of My Head '' focuses on an `` unhealthy '' and potentially destructive obsession .
He also noted that in the former song , Minogue played `` the wide - eyed ingénue with alacrity , '' but in the latter song she is aware of the harmful nature
of her infatuation , calling it a `` desire that is wholly dependent on her own self - control . ''
</item 10>
=== END ITEM id=c1579870cdbe7f15 ===

=== ITEM id=a42c1e2dec66b824 ===
Gold sentences and facts:
Sentence 0: No . overall No. in season Title Directed by Written by Original air date U.S. viewers ( millions ) `` Pilot '' Adam Davidson Robert Kirkman & Dave Erickson August 23 , 2015 ( 2015 - 08 - 23 ) 10.13 Nick awakens in a heroin den in an abandoned church , to find his girlfriend Gloria eating a corpse . While fleeing , he is hit by a car and hospitalized . The doctor tells Madison and Travis that Nick 's claims about the incident are heroin hallucinations , but Travis believes Nick after visiting the church himself . Alicia becomes more worried about Nick 's chemical dependency . The next day , school closes early due to the high levels of absenteeism and rumors of an epidemic . Nick escapes from the hospital and meets with Calvin , hoping to learn if the drugs Calvin sold him caused him to hallucinate in the church . Calvin tries to kill Nick to prevent him from exposing Calvin as a drug dealer or impugning the quality of his drugs . In the ensuing struggle , Calvin is mortally shot . After Travis and Madison arrive , the zombified Calvin attacks them . Nick runs over Calvin repeatedly with Travis ' truck , and the three watch in disbelief as the mutilated Calvin is still able to turn his head towards them . `` So Close , Yet So Far '' Adam Davidson Marco Ramirez August 30 , 2015 ( 2015 - 08 - 30 ) 8.18 After the run - in with zombified Calvin , Nick , Madison , and Travis choose to flee to the desert , and they want Alicia , Liza , and Chris to follow . Alicia finds Matt ill in his disheveled house . Travis arrives and sees a bite on Matt , who convinces Alicia to leave . The group returns to Madison 's home to gather supplies . Nick suffers from heroin withdrawal , so Madison drives to her school to get him Oxycodone . There , she encounters Tobias scavenging food . A zombified Artie tries to bite Tobias , so Madison kills Artie and drives Tobias home . Chris ' bus is trapped in a traffic jam caused by a zombie shooting by the police . He films the event and joins in a protest against that and other recent fatal police shootings , when Travis and Liza meet up with him . A riot erupts after police shoot down another zombie , but the three Manawas find refuge with the Salazars in their gated barbershop - home . Travis tells Madison to take the kids to the desert without him ; he will catch up . Alicia witnesses her zombified neighbor , Mr. Dawson , attacking Mrs. Cruz across the street , but Madison prevents her from intervening . The group inside the barbershop remains trapped , while the riot outside intensifies . `` The Dog '' Adam Davidson Jack LoGiudice September 13 , 2015 ( 2015 - 09 - 13 ) 7.19 While a riot rages outside , a mob sets fire to the store adjoining the barbershop , forcing the Salazars and Manawas to flee . The group reaches Travis ' truck and escapes , but not before Griselda is injured by a collapsing scaffold . Unable to reach a hospital , the group drives to Madison 's house , where Nick , Madison , and Alicia temporarily flee when the zombified Mr. Dawson attempts to enter , attracted by the barking dog Nick had let in . Nick leads Madison and Alicia to the Trans ' house next door , where they take a shotgun . Travis arrives and is attacked by Mr. Dawson , who is shot and killed by Daniel . All three families decide to stay the night and evacuate in the morning . Nurse Liza tends to Griselda 's injured foot but notes that Griselda will die if not treated by a doctor . Ofelia tells Daniel they should flee with Travis , but Daniel insists his family can survive alone and will join his cousin later . The next morning , as the Clarks and Manawas start driving away , the National Guard arrives and quarantines the block . While Travis says , `` It 's going to get better , '' Daniel laments that it 's , `` too late , '' as he watches a guardsman mark the neighboring house . `` Not Fade Away '' Kari Skogland Meaghan Oppenheimer September 20 , 2015 ( 2015 - 09 - 20 ) 6.62 Days after the National Guard quarantines the neighborhood into a Safe Zone , residents try to live normally . Tensions build under the military rule . Madison is stressed by extra work caused by her home 's being overcrowded and Travis ' new role as civilian liaison . Chris shows a video to Travis and Madison of a light signaling from the Dead Zone . Travis talks Doug into getting mental help . Liza medically helps neighbors . Nick steals morphine from Hector via IV drip . Ofelia kisses Adams , who was unable to get Griselda 's medicine . Madison exchanges signals with the flashing light in the Dead Zone . She sneaks outside the fence to investigate and finds evidence that the guardsmen killed civilians , even the uninfected . Travis learns that Doug has been hospitalized for his mental issues . Dr. Exner determines that Liza is not technically a nurse . Daniel warns Madison of his El Salvador experience , when the sick were taken under the guise of receiving hospitalization but instead killed . Soldiers take Griselda and Nick to a hospital , but Nick 's family protests his departure . Liza agrees to go to assist the medical team , despite not wanting to leave her son . Travis retreats to the roof and sees the signal from the Dead Zone . Seconds later , he sees and hears gunfire , followed by darkness . 5 5 `` Cobalt '' Kari Skogland David Wiener September 27 , 2015 ( 2015 - 09 - 27 ) 6.66 In a military cell , Strand bribes a guardsman to save feverish Nick from being moved . Liza helps Dr. Exner with patients at the hospital . Chris is devastated that Liza left voluntarily to help at the hospital , but Travis promises to bring her back . Madison discovers Daniel detaining Adams in the Trans ' basement . Alicia and Chris get drunk in and vandalize the abandoned home of a wealthy family . Strand recruits Nick for an escape plot . Travis convinces Moyers ' squad to take him to the hospital to check on his friends . While en route , Moyers encourages Travis to shoot a zombie , but Travis is emotionally unable to pull the trigger . The soldiers stop to assist another squad in a building infested by zombies , and most of those soldiers , including Moyers , are overcome . The few survivors flee and drop off Travis near the Safe Zone . Travis learns that Daniel tortured Adams into revealing what `` Cobalt '' means : in the morning , all civilians will be killed , and the guardsmen will evacuate the city . Griselda dies of septic shock at the hospital ; Liza shoots her brain to prevent reanimation . Daniel visits a nearby sports arena to verify Adams ' story that it was sealed with 2,000 now - zombified civilians inside . 6 6 `` The Good Man '' Stefan Schwartz Robert Kirkman & Dave Erickson October 4 , 2015 ( 2015 - 10 - 04 ) 6.86 The group drives to the National Guard 's headquarters to rescue Liza , Griselda , and Nick . Adams agrees to be their guide when let go by Travis . The group infiltrates the base after Daniel distracts the guards by leading a horde of walkers from the arena . Travis , Madison , Daniel , and Ofelia go inside , while Alicia and Chris stay behind . Meanwhile , the walkers breach the perimeter defenses and swarm the base . Travis ' group reach the holding cells and set the detainees free before reuniting with Nick , Liza , and Strand . They try to escape through the medical ward , where they discover Dr. Exner has euthanized all of the patients . Dr. Exner tells them of an escape route before presumably committing suicide . Before they can escape , the group encounters Adams , who shoots Ofelia in the arm . Enraged , Travis brutally beats Adams and leaves him for dead . Strand leads the group to his oceanside mansion , where he reveals to Nick that he owns a yacht which he plans to escape on , called the Abigail . On the beach , Liza reveals to Madison that she had been bitten during the escape . Liza pleads with Madison and Travis to euthanize her before she turns . Travis promises to protect Chris before shooting Liza .
  Fact 0.0: The table lists each episode's overall number, season number, title, director, writer, original air date, and U.S. viewers in millions.
  Fact 0.1: Episode 'Pilot' was directed by Adam Davidson and written by Robert Kirkman and Dave Erickson.
  Fact 0.2: Episode 'Pilot' originally aired on August 23, 2015.
  Fact 0.3: Episode 'Pilot' had 10.13 million U.S. viewers.
  Fact 0.4: Nick awakens in a heroin den in an abandoned church, to find his girlfriend Gloria eating a corpse.
  Fact 0.5: While fleeing, Nick is hit by a car and hospitalized.
  Fact 0.6: The doctor tells Madison and Travis that Nick's claims about the incident are heroin hallucinations.
  Fact 0.7: Travis believes Nick after Travis visits the church himself.
  Fact 0.8: Alicia becomes more worried about Nick's chemical dependency.
  Fact 0.9: The next day, school closes early due to high levels of absenteeism and rumors of an epidemic.
  Fact 0.10: Nick escapes from the hospital and meets with Calvin, hoping to learn if the drugs Calvin sold him caused him to hallucinate in the church.
  Fact 0.11: Calvin tries to kill Nick to prevent him from exposing Calvin as a drug dealer or impugning the quality of his drugs.
  Fact 0.12: In the ensuing struggle, Calvin is mortally shot.
  Fact 0.13: After Travis and Madison arrive, the zombified Calvin attacks them.
  Fact 0.14: Nick runs over Calvin repeatedly with Travis's truck.
  Fact 0.15: The three watch in disbelief as the mutilated Calvin is still able to turn his head towards them.
  Fact 0.16: Episode 'So Close, Yet So Far' was directed by Adam Davidson and written by Marco Ramirez.
  Fact 0.17: Episode 'So Close, Yet So Far' originally aired on August 30, 2015.
  Fact 0.18: Episode 'So Close, Yet So Far' had 8.18 million U.S. viewers.
  Fact 0.19: After the run-in with zombified Calvin, Nick, Madison, and Travis choose to flee to the desert.
  Fact 0.20: Nick, Madison, and Travis want Alicia, Liza, and Chris to follow them to the desert.
  Fact 0.21: Alicia finds Matt ill in his disheveled house.
  Fact 0.22: Travis arrives and sees a bite on Matt.
  Fact 0.23: Matt convinces Alicia to leave.
  Fact 0.24: The group returns to Madison's home to gather supplies.
  Fact 0.25: Nick suffers from heroin withdrawal, so Madison drives to her school to get him Oxycodone.
  Fact 0.26: At the school, Madison encounters Tobias scavenging food.
  Fact 0.27: A zombified Artie tries to bite Tobias, so Madison kills Artie and drives Tobias home.
  Fact 0.28: Chris's bus is trapped in a traffic jam caused by a zombie shooting by the police.
  Fact 0.29: Chris films the event and joins in a protest against that and other recent fatal police shootings, when Travis and Liza meet up with him.
  Fact 0.30: A riot erupts after police shoot down another zombie.
  Fact 0.31: The three Manawas find refuge with the Salazars in their gated barbershop-home.
  Fact 0.32: Travis tells Madison to take the kids to the desert without him, and he will catch up.
  Fact 0.33: Alicia witnesses her zombified neighbor, Mr. Dawson, attacking Mrs. Cruz across the street.
  Fact 0.34: Madison prevents Alicia from intervening.
  Fact 0.35: The group inside the barbershop remains trapped, while the riot outside intensifies.
  Fact 0.36: Episode 'The Dog' was directed by Adam Davidson and written by Jack LoGiudice.
  Fact 0.37: Episode 'The Dog' originally aired on September 13, 2015.
  Fact 0.38: Episode 'The Dog' had 7.19 million U.S. viewers.
  Fact 0.39: While a riot rages outside, a mob sets fire to the store adjoining the barbershop, forcing the Salazars and Manawas to flee.
  Fact 0.40: The group reaches Travis's truck and escapes, but not before Griselda is injured by a collapsing scaffold.
  Fact 0.41: Unable to reach a hospital, the group drives to Madison's house.
  Fact 0.42: Nick, Madison, and Alicia temporarily flee when the zombified Mr. Dawson attempts to enter.
  Fact 0.43: Mr. Dawson is attracted by the barking dog Nick had let in.
  Fact 0.44: Nick leads Madison and Alicia to the Trans' house next door, where they take a shotgun.
  Fact 0.45: Travis arrives and is attacked by Mr. Dawson.
  Fact 0.46: Mr. Dawson is shot and killed by Daniel.
  Fact 0.47: All three families decide to stay the night and evacuate in the morning.
  Fact 0.48: Nurse Liza tends to Griselda's injured foot but notes that Griselda will die if not treated by a doctor.
  Fact 0.49: Ofelia tells Daniel they should flee with Travis.
  Fact 0.50: Daniel insists his family can survive alone and will join his cousin later.
  Fact 0.51: The next morning, as the Clarks and Manawas start driving away, the National Guard arrives and quarantines the block.
  Fact 0.52: Travis says 'It's going to get better.'
  Fact 0.53: Daniel laments that it's 'too late' as he watches a guardsman mark the neighboring house.
  Fact 0.54: Episode 'Not Fade Away' was directed by Kari Skogland and written by Meaghan Oppenheimer.
  Fact 0.55: Episode 'Not Fade Away' originally aired on September 20, 2015.
  Fact 0.56: Episode 'Not Fade Away' had 6.62 million U.S. viewers.
  Fact 0.57: Days after the National Guard quarantines the neighborhood into a Safe Zone, residents try to live normally.
  Fact 0.58: Tensions build under the military rule.
  Fact 0.59: Madison is stressed by extra work caused by her home's being overcrowded.
  Fact 0.60: Madison is stressed by Travis's new role as civilian liaison.
  Fact 0.61: Chris shows a video to Travis and Madison of a light signaling from the Dead Zone.
  Fact 0.62: Travis talks Doug into getting mental help.
  Fact 0.63: Liza medically helps neighbors.
  Fact 0.64: Nick steals morphine from Hector via IV drip.
  Fact 0.65: Ofelia kisses Adams, who was unable to get Griselda's medicine.
  Fact 0.66: Madison exchanges signals with the flashing light in the Dead Zone.
  Fact 0.67: Madison sneaks outside the fence to investigate.
  Fact 0.68: Madison finds evidence that the guardsmen killed civilians, even the uninfected.
  Fact 0.69: Travis learns that Doug has been hospitalized for his mental issues.
  Fact 0.70: Dr. Exner determines that Liza is not technically a nurse.
  Fact 0.71: Daniel warns Madison of his El Salvador experience, when the sick were taken under the guise of receiving hospitalization but instead killed.
  Fact 0.72: Soldiers take Griselda and Nick to a hospital, but Nick's family protests his departure.
  Fact 0.73: Liza agrees to go to assist the medical team, despite not wanting to leave her son.
  Fact 0.74: Travis retreats to the roof and sees the signal from the Dead Zone.
  Fact 0.75: Seconds later, Travis sees and hears gunfire, followed by darkness.
  Fact 0.76: Episode 'Cobalt' is overall number 5 and season number 5.
  Fact 0.77: Episode 'Cobalt' was directed by Kari Skogland and written by David Wiener.
  Fact 0.78: Episode 'Cobalt' originally aired on September 27, 2015.
  Fact 0.79: Episode 'Cobalt' had 6.66 million U.S. viewers.
  Fact 0.80: In a military cell, Strand bribes a guardsman to save feverish Nick from being moved.
  Fact 0.81: Liza helps Dr. Exner with patients at the hospital.
  Fact 0.82: Chris is devastated that Liza left voluntarily to help at the hospital.
  Fact 0.83: Travis promises to bring Liza back.
  Fact 0.84: Madison discovers Daniel detaining Adams in the Trans' basement.
  Fact 0.85: Alicia and Chris get drunk in and vandalize the abandoned home of a wealthy family.
  Fact 0.86: Strand recruits Nick for an escape plot.
  Fact 0.87: Travis convinces Moyers' squad to take him to the hospital to check on his friends.
  Fact 0.88: While en route, Moyers encourages Travis to shoot a zombie.
  Fact 0.89: Travis is emotionally unable to pull the trigger.
  Fact 0.90: The soldiers stop to assist another squad in a building infested by zombies.
  Fact 0.91: Most of those soldiers, including Moyers, are overcome.
  Fact 0.92: The few survivors flee and drop off Travis near the Safe Zone.
  Fact 0.93: Travis learns that Daniel tortured Adams into revealing what 'Cobalt' means.
  Fact 0.94: 'Cobalt' means that in the morning, all civilians will be killed, and the guardsmen will evacuate the city.
  Fact 0.95: Griselda dies of septic shock at the hospital.
  Fact 0.96: Liza shoots Griselda's brain to prevent reanimation.
  Fact 0.97: Daniel visits a nearby sports arena to verify Adams' story that it was sealed with 2,000 now-zombified civilians inside.
  Fact 0.98: Episode 'The Good Man' is overall number 6 and season number 6.
  Fact 0.99: Episode 'The Good Man' was directed by Stefan Schwartz and written by Robert Kirkman and Dave Erickson.
  Fact 0.100: Episode 'The Good Man' originally aired on October 4, 2015.
  Fact 0.101: Episode 'The Good Man' had 6.86 million U.S. viewers.
  Fact 0.102: The group drives to the National Guard's headquarters to rescue Liza, Griselda, and Nick.
  Fact 0.103: Adams agrees to be their guide when let go by Travis.
  Fact 0.104: The group infiltrates the base after Daniel distracts the guards by leading a horde of walkers from the arena.
  Fact 0.105: Travis, Madison, Daniel, and Ofelia go inside, while Alicia and Chris stay behind.
  Fact 0.106: Meanwhile, the walkers breach the perimeter defenses and swarm the base.
  Fact 0.107: Travis's group reaches the holding cells and sets the detainees free.
  Fact 0.108: Travis's group reunites with Nick, Liza, and Strand before trying to escape.
  Fact 0.109: The group tries to escape through the medical ward.
  Fact 0.110: The group discovers Dr. Exner has euthanized all of the patients in the medical ward.
  Fact 0.111: Dr. Exner tells them of an escape route before presumably committing suicide.
  Fact 0.112: Before they can escape, the group encounters Adams, who shoots Ofelia in the arm.
  Fact 0.113: Enraged, Travis brutally beats Adams and leaves him for dead.
  Fact 0.114: Strand leads the group to his oceanside mansion.
  Fact 0.115: At the mansion, Strand reveals to Nick that he owns a yacht which he plans to escape on, called the Abigail.
  Fact 0.116: On the beach, Liza reveals to Madison that she had been bitten during the escape.
  Fact 0.117: Liza pleads with Madison and Travis to euthanize her before she turns.
  Fact 0.118: Travis promises to protect Chris before shooting Liza.

Retrieved items:
<item 1>
Fear the Walking Dead (season 1)
`` Not Fade Away '' Kari Skogland Meaghan Oppenheimer September 20 , 2015 ( 2015 - 09 - 20 ) 6.62 Days after the National Guard quarantines the neighborhood
into a Safe Zone , residents try to live normally . Tensions build under the military rule . Madison is stressed by extra work caused by her home 's being
overcrowded and Travis ' new role as civilian liaison . Chris shows a video to Travis and Madison of a light signaling from the Dead Zone . Travis talks Doug
into getting mental help . Liza medically helps neighbors . Nick steals morphine from Hector via IV drip . Ofelia kisses Adams , who was unable to get Griselda
's medicine . Madison exchanges signals with the flashing light in the Dead Zone . She sneaks outside the fence to investigate and finds evidence that the
guardsmen killed civilians , even the uninfected . Travis learns that Doug has been hospitalized for his mental issues . Dr. Exner determines that Liza is not
technically a nurse .
</item 1>

<item 2>
Fear the Walking Dead (season 1)
1 Talking Dead Fear the Walking Dead Episodes season 1 Web series The Walking Dead Torn Apart Cold Storage Fear the Walking Dead Flight 462 Miscellaneous Video
games Season One `` 400 Days '' Two Michonne Three : A New Frontier Other games Survival Instinct Road to Survival Rides The Walking Dead : The Ride Novels Rise
of the Governor The Road to Woodbury The Fall of the Governor Descent Invasion Search and Destroy Return to Woodbury Characters Lists Comic book series
Television series Aaron Alden Amy Jessie Anderson Andrea Axel Tara Chambler Clementine Daryl Dixon Merle Dixon Dwight Rosita Espinosa Lee Everett King Ezekiel
Abraham Ford Gareth The Governor Gregory Beth Greene Hershel Greene Maggie Greene Carl Grimes Lori Grimes Rick Grimes Dale Horvath Jadis Morgan Jones Lilly
Milton Mamet Caesar Martinez Deanna Monroe Paul `` Jesus '' Monroe Spencer Monroe Morales Michonne Negan Olivia The Prisoners Carol Peletier Sophia Peletier
Eugene Porter Glenn Rhee Lizzie and Mika Samuels Simon Gabriel Stokes Bob Stookey T - Dog Shane Walsh Sasha Williams Tyreese Williams Retrieved from ``
https://en.wikipedia.org/w/index.
</item 2>

<item 3>
Fear the Walking Dead (season 1)
Alicia and Chris get drunk in and vandalize the abandoned home of a wealthy family . Strand recruits Nick for an escape plot . Travis convinces Moyers ' squad
to take him to the hospital to check on his friends . While en route , Moyers encourages Travis to shoot a zombie , but Travis is emotionally unable to pull the
trigger . The soldiers stop to assist another squad in a building infested by zombies , and most of those soldiers , including Moyers , are overcome . The few
survivors flee and drop off Travis near the Safe Zone . Travis learns that Daniel tortured Adams into revealing what `` Cobalt '' means : in the morning , all
civilians will be killed , and the guardsmen will evacuate the city . Griselda dies of septic shock at the hospital ; Liza shoots her brain to prevent
reanimation . Daniel visits a nearby sports arena to verify Adams ' story that it was sealed with 2,000 now - zombified civilians inside .
</item 3>

<item 4>
Fear the Walking Dead (season 1)
Fear the Walking Dead ( season 1 ) - wikipedia Fear the Walking Dead ( season 1 ) Jump to : navigation , search Fear the Walking Dead ( season 1 ) Home media
cover art Starring Kim Dickens Cliff Curtis Frank Dillane Alycia Debnam - Carey Elizabeth Rodriguez Mercedes Mason Lorenzo James Henrie Rubén Blades Country of
origin United States No. of episodes 6 Release Original network AMC Original release August 23 ( 2015 - 08 - 23 ) -- October 4 , 2015 ( 2015 - 10 - 04 ) Season
chronology Next → Season 2 List of Fear the Walking Dead episodes The first season of Fear the Walking Dead , an American horror - drama television series on
AMC , premiered on August 23 , 2015 , and concluded on October 4 , 2015 , consisting of six episodes . The series is a companion series and prequel to The
Walking Dead , which is based on the comic book series of the same name by Robert Kirkman , Tony Moore , and Charlie Adlard .
</item 4>

<item 5>
Fear the Walking Dead (season 1)
6 6 `` The Good Man '' Stefan Schwartz Robert Kirkman & Dave Erickson October 4 , 2015 ( 2015 - 10 - 04 ) 6.86 The group drives to the National Guard 's
headquarters to rescue Liza , Griselda , and Nick . Adams agrees to be their guide when let go by Travis . The group infiltrates the base after Daniel distracts
the guards by leading a horde of walkers from the arena . Travis , Madison , Daniel , and Ofelia go inside , while Alicia and Chris stay behind . Meanwhile ,
the walkers breach the perimeter defenses and swarm the base . Travis ' group reach the holding cells and set the detainees free before reuniting with Nick ,
Liza , and Strand . They try to escape through the medical ward , where they discover Dr. Exner has euthanized all of the patients . Dr. Exner tells them of an
escape route before presumably committing suicide . Before they can escape , the group encounters Adams , who shoots Ofelia in the arm . Enraged , Travis
brutally beats Adams and leaves him for dead .
</item 5>

<item 6>
Fear the Walking Dead (season 1)
Rotten Tomatoes . Retrieved August 25 , 2015 . Jump up ^ `` Fear the Walking Dead : Season 1 '' . Metacritic . Retrieved August 25 , 2015 . Jump up ^
Vincentelli , Elisabeth ( August 20 , 2015 ) . `` ' Walking Dead ' prequel is creepily suspenseful '' . New York Post . Retrieved August 20 , 2015 . Jump up ^
Tucker , Ken ( August 5 , 2015 ) . `` ' Fear the Walking Dead ' : Artful Suspense , Not Bloody Horror '' . Yahoo TV . Retrieved August 20 , 2015 . Jump up ^
Goodman , Tim ( August 4 , 2015 ) . `` ' Fear the Walking Dead ' : TV Review '' . The Hollywood Reporter . Retrieved August 20 , 2015 . Jump up ^ Fienberg ,
Daniel ( August 20 , 2015 ) . `` No. 296 - ' Fear the Walking Dead ' and more '' . HitFix .
</item 6>

<item 7>
Fear the Walking Dead (season 1)
TV by the Numbers . October 2 , 2015 . Retrieved March 27 , 2018 . Jump up ^ Lambert , David ( October 6 , 2015 ) . `` Fear the Walking Dead - Press Release
Announces ' The Complete 1st Season ' on Blu - ray , DVD '' . TVShowsOnDVD.com . Retrieved October 6 , 2015 . Jump up ^ Lambert , David ( January 20 , 2016 ) .
`` Fear the Walking Dead - ' The Complete 1st Season : Special Edition 's ' Press Release ! '' . TVShowsOnDVD.com . Retrieved January 21 , 2016 . External links
( edit ) Official website List of Fear the Walking Dead episodes on IMDb ( hide ) The Walking Dead Robert Kirkman Tony Moore Charlie Adlard Comic books The
Walking Dead Television series The Walking Dead Episodes season 1 5 6 7 8 Accolades Soundtrack Vol.
</item 7>

<item 8>
Fear the Walking Dead (season 1)
overall No. in season Title Directed by Written by Original air date U.S. viewers ( millions ) `` Pilot '' Adam Davidson Robert Kirkman & Dave Erickson August
23 , 2015 ( 2015 - 08 - 23 ) 10.13 Nick awakens in a heroin den in an abandoned church , to find his girlfriend Gloria eating a corpse . While fleeing , he is
hit by a car and hospitalized . The doctor tells Madison and Travis that Nick 's claims about the incident are heroin hallucinations , but Travis believes Nick
after visiting the church himself . Alicia becomes more worried about Nick 's chemical dependency . The next day , school closes early due to the high levels of
absenteeism and rumors of an epidemic . Nick escapes from the hospital and meets with Calvin , hoping to learn if the drugs Calvin sold him caused him to
hallucinate in the church . Calvin tries to kill Nick to prevent him from exposing Calvin as a drug dealer or impugning the quality of his drugs . In the
ensuing struggle , Calvin is mortally shot . After Travis and Madison arrive , the zombified Calvin attacks them .
</item 8>

<item 9>
Fear the Walking Dead (season 1)
3 6.62 2.0 3.75 5.3 10.37 5 `` Cobalt '' September 27 , 2015 3.4 6.66 1.8 3.37 5.3 10.03 6 `` The Good Man '' October 4 , 2015 3.4 6.86 N / A N / A N / A N / A
Home media releases ( edit ) The first season was released on Blu - ray and DVD on December 1 , 2015 . A special edition version of the first season was
released on Blu - ray and DVD on March 22 , 2016 , with new bonus features , including deleted scenes , seven featurettes , and audio commentaries by cast and
crew , on all six episodes . References ( edit ) Jump up ^ Prudom , Laura ( August 5 , 2015 ) . `` ' Fear the Walking Dead ' Cast and Creators Talk Kickstarting
the Apocalypse '' . Variety . Retrieved August 12 , 2015 .
</item 9>

<item 10>
Fear the Walking Dead (season 1)
Chris ' bus is trapped in a traffic jam caused by a zombie shooting by the police . He films the event and joins in a protest against that and other recent
fatal police shootings , when Travis and Liza meet up with him . A riot erupts after police shoot down another zombie , but the three Manawas find refuge with
the Salazars in their gated barbershop - home . Travis tells Madison to take the kids to the desert without him ; he will catch up . Alicia witnesses her
zombified neighbor , Mr. Dawson , attacking Mrs. Cruz across the street , but Madison prevents her from intervening . The group inside the barbershop remains
trapped , while the riot outside intensifies . `` The Dog '' Adam Davidson Jack LoGiudice September 13 , 2015 ( 2015 - 09 - 13 ) 7.19 While a riot rages outside
, a mob sets fire to the store adjoining the barbershop , forcing the Salazars and Manawas to flee . The group reaches Travis ' truck and escapes , but not
before Griselda is injured by a collapsing scaffold .
</item 10>
=== END ITEM id=a42c1e2dec66b824 ===
