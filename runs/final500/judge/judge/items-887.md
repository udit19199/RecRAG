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

=== ITEM id=7940b3c4260de05e ===
Gold sentences and facts:
Sentence 0: All I Want for Christmas Is You , a Night of Joy and Festivity is the first concert residency by American singer Mariah Carey . Performed annually at the Beacon Theatre in Manhattan , New York , the residency began on December 15 , 2014 , and will conclude on December 22 , 2017 , after completing six legs and thirty - four shows . Billboard revealed that the six dates had sold out with a total of 16,196 tickets while grossing $1.5 million during the first leg of the residency . The main set list for the show encompassed songs from Merry Christmas and Merry Christmas II You , with additional non-holiday tracks . The show also included `` a cast of forty , including a gospel choir , a clutch of carolers , several acrobats and a hyperactive round of kid singer / dancers . ''
  Fact 0.0: 'All I Want for Christmas Is You, a Night of Joy and Festivity' is the first concert residency by American singer Mariah Carey.
  Fact 0.1: The residency is performed annually at the Beacon Theatre in Manhattan, New York.
  Fact 0.2: The residency began on December 15, 2014.
  Fact 0.3: The residency will conclude on December 22, 2017.
  Fact 0.4: The residency completes six legs and thirty-four shows.
  Fact 0.5: Billboard revealed that the six dates had sold out with a total of 16,196 tickets.
  Fact 0.6: The six dates grossed $1.5 million during the first leg of the residency.
  Fact 0.7: The main set list for the show encompassed songs from Merry Christmas and Merry Christmas II You.
  Fact 0.8: The main set list for the show included additional non-holiday tracks.
  Fact 0.9: The show included a cast of forty.
  Fact 0.10: The show's cast included a gospel choir, a clutch of carolers, several acrobats, and a hyperactive round of kid singer/dancers.

Retrieved items:
<item 1>
All I Want for Christmas Is You, a Night of Joy and Festivity
Jump up ^ Gamboa , Glenn ( October 31 , 2014 ) . `` Mariah Carey sets holiday shows at Beacon Theatre '' . Newsday . Retrieved January 28 , 2015 . ^ Jump up to
: Farber , Jim ( December 15 , 2014 ) . `` Mariah Carey overdoes herself at wavering Beacon Theatre show : concert review '' . Daily News . New York . Retrieved
January 28 , 2015 . ^ Jump up to : Stern , Bradley ( August 18 , 2014 ) . `` Mariah Carey Announces ' The Elusive Chanteuse Show ' : See The First Tour Dates ''
. Idolator . Archived from the original on October 6 , 2017 . Retrieved October 6 , 2017 . ^ Jump up to : Brandle , Lars ( September 18 , 2014 ) . `` Mariah
Carey Announces ' Elusive Chanteuse ' Tour of Australia '' . Billboard . Archived from the original on October 6 , 2017 .
</item 1>

<item 2>
Mariah Carey’s first concert residency, “All I Want for Christmas Is You, a Night of Joy and Festivity,” ran annually from 2014–17, primarily at New York’s
Beacon Theatre, with 34 shows across six legs. Featuring Christmas favorites, other hits, choir, carolers and dancers, it grossed about $7.34 million; several
2017 dates were canceled because Carey had a respiratory infection.

“All I Want for Christmas Is My Two Front Teeth,”
</item 2>

<item 3>
All I Want for Christmas Is You, a Night of Joy and Festivity
of shows 6 in 2014 8 in 2015 9 in 2016 11 in 2017 34 in total Attendance 65,195 ( 93 % ) Box office $7,335,589 Mariah Carey concert chronology The Elusive
Chanteuse Show ( 2014 ) All I Want for Christmas Is You , a Night of Joy & Festivity ( 2014 - 17 ) Number 1 to Infinity ( 2015 -- 17 ) All I Want for Christmas
Is You , a Night of Joy and Festivity is the first concert residency by American singer Mariah Carey . Performed annually at the Beacon Theatre in Manhattan ,
New York , the residency began on December 15 , 2014 , and will conclude on December 22 , 2017 , after completing six legs and thirty - four shows . Billboard
revealed that the six dates had sold out with a total of 16,196 tickets while grossing $1.5 million during the first leg of the residency .
</item 3>

<item 4>
All I Want for Christmas Is You, a Night of Joy and Festivity
Jump up ^ http://www.torontosun.com/2017/07/19/mariah-carey-brings-christmas-concert-series-to-las-vegas Jump up ^
http://windsorstar.com/news/local-news/mariah-carey-cancels-caesars-windsor-show Jump up ^
http://www.etonline.com/mariah-carey-cancels-first-shows-christmas-concert-due-respiratory-infection-91147 Jump up ^
http://www.billboard.com/articles/columns/pop/8038463/mariah-carey-cancels-shows-christmas-tour Jump up ^
http://metro.co.uk/2017/11/25/mariah-carey-cancels-more-christmas-tour-shows-for-a-few-extra-days-rest-7107422/ Jump up ^ http://www.eonline.
</item 4>

<item 5>
All I Want for Christmas Is You, a Night of Joy and Festivity
All I Want for Christmas is You , a Night of Joy and Festivity - wikipedia All I Want for Christmas is You , a Night of Joy and Festivity Jump to : navigation ,
search All I Want for Christmas Is You , a Night of Joy and Festivity Residency by Mariah Carey Location North America Europe Venue Various Associated albums
Merry Christmas & Merry Christmas II You Start date December 15 , 2014 ( 2014 - 12 - 15 ) End date December 22 , 2017 ( 2017 - 12 - 22 ) Legs 6 No.
</item 5>

<item 6>
All I Want for Christmas Is You, a Night of Joy and Festivity
I Am Mariah ... The Elusive Chanteuse Live albums MTV Unplugged Compilation albums # 1 's Greatest Hits The Remixes The Ballads The Essential Mariah Carey # 1
to Infinity Soundtrack albums Glitter Video albums The First Vision MTV Unplugged + 3 Here Is Mariah Carey Fantasy : Mariah Carey at Madison Square Garden
Around the World # 1 's The Adventures of Mimi Television VH1 Divas American Idol A Christmas Melody Mariah 's World Concert tours Music Box Tour Daydream World
Tour Butterfly World Tour Rainbow World Tour Charmbracelet World Tour The Adventures of Mimi Angels Advocate Tour The Elusive Chanteuse Show The Sweet Sweet
Fantasy Tour All the Hits Tour The # 1 's Tour Concert residencies Live at the Pearl All I Want for Christmas Is You , a Night of Joy and Festivity # 1 to
Infinity Fragrances M by Mariah Carey Luscious Pink Forever Companies Monarc Entertainment Butterfly MC Records Related topics Tommy Mottola Nick Cannon Lionel
Cole James Packer VH1 Divas Book Portal Retrieved from `` https://en.wikipedia.org/w/index.
</item 6>

<item 7>
All I Want for Christmas Is You, a Night of Joy and Festivity
eonline.com/news/896163/mariah-carey-further-postpones-christmas-tour-over-health-issues Book : Merry Christmas II You ( hide ) Mariah Carey Albums Songs Awards
and nominations Singles Videography Performances Studio albums Mariah Carey Emotions Music Box Merry Christmas Daydream Butterfly Rainbow Charmbracelet The
Emancipation of Mimi E = MC2 Memoirs of an Imperfect Angel Merry Christmas II You Me . I Am Mariah ...
</item 7>

<item 8>
All I Want for Christmas Is You, a Night of Joy and Festivity
wikipedia.org/w/index.php?title=All_I_Want_for_Christmas_Is_You,_a_Night_of_Joy_and_Festivity&oldid=819618785 '' Categories : 2014 concert residencies 2015
concert residencies 2016 concert residencies 2017 concert residencies Mariah Carey concert residencies Christmas music Hidden categories : Pages using div col
with deprecated parameters Talk Contents About Wikipedia Español Bahasa Indonesia Português Edit links This page was last edited on 10 January 2018 , at 10 : 27
. Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of
Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 8>

<item 9>
All I Want for Christmas Is You, a Night of Joy and Festivity
195 / 74,089 ( 93 % ) $7,335,589 Cancelled shows ( edit ) Date Venue City Country Reason 2017 November 17 , 2017 Caesars Windsor Windsor Canada `` Artist
illness '' ( Upper respiratory infection ) November 18 , 2017 Chicago Theatre Chicago United States November 20 , 2017 Sands Bethlehem Event Center Bethlehem
November 21 , 2017 Seneca Niagara Events Center Niagara Falls November 22 , 2017 Hard Rock Rocksino Northfield Park Cleveland November 24 , 2017 MGM National
Harbor Washington , D.C. November 25 , 2017 Borgata Hotel Casino & Spa Atlantic City November 28 , 2017 Beacon Theatre New York City December 1 , 2017
References ( edit ) Jump up ^ Lopez , Francisco ( October 31 , 2014 ) . `` Mariah Carey 's ' All I Want For Christmas Is You ' performance at the Beacon NYC ''
. AXS.com . Retrieved January 28 , 2015 . Jump up ^ Gamboa , Glenn ( October 31 , 2014 ) .
</item 9>

<item 10>
All I Want for Christmas Is You, a Night of Joy and Festivity
The main set list for the show encompassed songs from Merry Christmas and Merry Christmas II You , with additional non-holiday tracks . The show also included
`` a cast of forty , including a gospel choir , a clutch of carolers , several acrobats and a hyperactive round of kid singer / dancers . '' Contents ( hide ) 1
Background and development 2 Critical reception 3 Commercial performance 4 Set list 5 Residency dates 5.1 Cancelled shows 6 References Background and
development ( edit ) Mariah Carey released her fourteenth studio album , titled Me . I Am Mariah ... The Elusive Chanteuse , on May 27 , 2014 . It debuted at
number three on the Billboard 200 album chart with opening week sales of 58,000 copies in the United States , and number one on the Top R&B / Hip - Hop Albums
chart . Carey embarked on her eighth concert tour called The Elusive Chanteuse Show in October and November later that year , whereby she performed concerts
throughout East Asia , South East Asia and Australasia .
</item 10>
=== END ITEM id=7940b3c4260de05e ===

=== ITEM id=f892d6dc2eb61990 ===
Gold sentences and facts:
Sentence 0: The Battle of Lake Erie , sometimes called the Battle of Put - in - Bay , was fought on 10 September 1813 , in Lake Erie off the coast of Ohio during the War of 1812 . Nine vessels of the United States Navy defeated and captured six vessels of the British Royal Navy . This ensured American control of the lake for the rest of the war , which in turn allowed the Americans to recover Detroit and win the Battle of the Thames to break the Indian confederation of Tecumseh . It was one of the biggest naval battles of the War of 1812 .
  Fact 0.0: The Battle of Lake Erie is sometimes called the Battle of Put-in-Bay.
  Fact 0.1: The Battle of Lake Erie was fought on 10 September 1813.
  Fact 0.2: The Battle of Lake Erie was fought in Lake Erie off the coast of Ohio.
  Fact 0.3: The Battle of Lake Erie was fought during the War of 1812.
  Fact 0.4: Nine vessels of the United States Navy defeated six vessels of the British Royal Navy in the Battle of Lake Erie.
  Fact 0.5: Nine vessels of the United States Navy captured six vessels of the British Royal Navy in the Battle of Lake Erie.
  Fact 0.6: The Battle of Lake Erie ensured American control of the lake for the rest of the War of 1812.
  Fact 0.7: The Battle of Lake Erie allowed the Americans to recover Detroit.
  Fact 0.8: The Battle of Lake Erie allowed the Americans to win the Battle of the Thames.
  Fact 0.9: Winning the Battle of the Thames broke the Indian confederation of Tecumseh.
  Fact 0.10: The Battle of Lake Erie was one of the biggest naval battles of the War of 1812.

Retrieved items:
<item 1>
Battle of Lake Erie
- 82.82500 Retrieved from `` https://en.wikipedia.org/w/index.php?title=Battle_of_Lake_Erie&oldid=799568331 '' Categories : Conflicts in 1813 Battles involving
Canada History of Ohio Naval battles of the War of 1812 Naval battles involving Canada Lake Erie 1813 in Ohio September 1813 events Hidden categories : CS1
maint : Extra text : authors list Pages using deprecated image syntax Articles needing additional references from September 2017 All articles needing additional
references Wikipedia articles incorporating a citation from The American Cyclopaedia Wikipedia articles incorporating a citation from The American Cyclopaedia
with a Wikisource reference Coordinates not on Wikidata Use dmy dates from May 2011 Talk Contents About Wikipedia Afrikaans Català Deutsch Español Français 한국어
Magyar Bahasa Melayu 日本 語 Polski Português ไทย 中文 Edit links This page was last edited on 8 September 2017 , at 14 : 22 . Text is available under the Creative
Commons Attribution - ShareAlike License ; additional terms may apply .
</item 1>

<item 2>
Battle of Lake Erie
USS United States vs HMS Macedonian Caribbean Sea Battle of La Guaira Great Lakes Battle of Lake Erie Battle of Fort Oswego Engagements on Lake Huron
Engagements on Lake Ontario First Battle of Sacket 's Harbor Gulf Coast Action of 13 December 1814 Battle of Lake Borgne Pacific Ocean Action off James Island
Action off Charles Island Nuku Hiva Campaign Battle of Valparaiso ( Capture of USS Essex ) See also : American Indian Wars , Creek War , Napoleonic Wars , and
Tecumseh 's War Portal definition textbooks quotes source texts media news stories Coordinates : 41 ° 39 ′ 44 '' N 82 ° 49 ′ 30 '' W ﻿ / ﻿ 41.66222 ° N 82.82500
° W ﻿ / 41.66222 ; - 82.82500 Retrieved from `` https://en.wikipedia.org/w/index.
</item 2>

<item 3>
Battle of Lake Erie
James Massacre Missouri Battle of Credit Island Battle of the Sink Hole British North America Lower Canada Battle of the Chateauguay First Battle of Lacolle
Mills Second Battle of Lacolle Mills Upper Canada Battle of Beaver Dams Battle of Chippawa Battle of Cook 's Mills Battle of Crysler 's Farm Battle of Fort
George Battle of Frenchman 's Creek Battle of Longwoods Battle of Lundy 's Lane Battle of Malcolm 's Mills Battle of Queenston Heights Battle of Stoney Creek
Battle of the Thames Battle of York Capture of Fort Erie Raid on Elizabethtown Raid on Port Dover Raid on Gananoque Siege of Fort Erie Spanish Empire Spanish
Florida Battle of Pensacola Naval battles Atlantic Ocean Capture of HMS Boxer Capture of HMS Cyane Capture of HMS Epervier Capture of HMS Frolic Capture of HMS
Penguin Capture of HMS Dominica Capture of USS Argus Capture of USS Chesapeake Capture of USS President Chesapeake Bay Flotilla USS Constitution vs HMS Java
Sinking of HMS Avon Battle of Fayal Sinking of HMS Peacock Sinking of HMS Reindeer USS Constitution vs HMS Guerriere USS United States vs HMS Macedonian
Caribbean Sea Battle of La Guaira Great Lakes Battle of Lake Erie
</item 3>

<item 4>
Battle of Lake Erie
ISBN 0 - 8117 - 2876 - 5 . Roosevelt , Theodore ( 2004 ) . The Naval War of 1812 . New York : The Modern Library . ISBN 0 - 375 - 75419 - 9 . Skaggs , David ;
Atloff , Gerard ( 1997 ) . A Signal Victory : The Lake Erie Campaign , 1812 -- 13 . Naval Institute Press . ISBN 1 - 55750 - 892 - 5 . Symonds , Craig ( 2005 )
. Decision at Sea . Oxford University Press . ISBN 0 - 19 - 517145 - 4 . Zaslow , Morris ( ed ) . ( 1964 ) . The Defended Border : Upper Canada and the War of
1812 . Toronto : Macmillan of Canada . ISBN 0 - 7705 - 1242 - 9 . Further reading ( edit ) Axelrod , Alen ; Phillips , Charles . The MacMillan Dictionary of
Military Biography ( New York : MacMillan , 1998 . )
</item 4>

<item 5>
Battle of Lake Erie
Naval campaigns of the War of 1812 Atlantic Ocean USS Essex vs HMS Alert USS Constitution vs HMS Guerriere Capture of HMS Frolic USS United States vs HMS
Macedonian USS Constitution vs HMS Java Sinking of HMS Peacock Rappahannock River Capture of USS Chesapeake Capture of HMS Dominica Capture of USS Argus Capture
of HMS Boxer Capture of USS Frolic Capture of HMS Epervier Sinking of HMS Reindeer Sinking of HMS Avon Fayal Capture of USS President Capture of HMS Cyane and
HMS Levant Capture of HMS Penguin Capture of East India Company ship Nautilus East Coast Chesapeake Bay Alexandria Baltimore Hampden Fort Peter Great Lakes /
Saint Lawrence River Lake Ontario 1st Sacket 's Harbor York Fort George 2nd Sacket 's Harbor Lake Erie Fort Oswego Lake Huron Lake Champlain West Indies / Gulf
Coast La Guaira 1st Fort Bowyer Action of 13 December 1814 Lake Borgne New Orleans Fort St.
</item 5>

<item 6>
Battle of Lake Erie
It was one of the biggest naval battles of the War of 1812 . Contents ( hide ) 1 Background 1.1 1812 1.2 1813 1.3 Blockades of Presque Isle and Amherstburg 2
Battle 3 Casualties 4 Aftermath 5 Reconstructions and memorials 6 Reasons for the American victory 7 Order of battle 8 Footnotes 9 Notes 10 References 11
Further reading 12 External links Background ( edit ) 1812 ( edit ) When the war broke out , the British immediately seized control of Lake Erie . They already
had a small force of warships there : the sloop - of - war Queen Charlotte and the brig General Hunter . The schooner Lady Prevost was under construction and
was put into service a few weeks after the outbreak of war . These vessels were controlled by the Provincial Marine , which was a military transport service and
not a naval service . Nevertheless , the Americans lacked any counter to the British armed vessels .
</item 6>

<item 7>
Battle of Lake Erie
External links ( edit ) Wikimedia Commons has media related to Battle of Lake Erie . United States Navy portal Rhode Island portal Erie portal Ohio portal War
of 1812 portal History of Canada portal Wikiquote has quotations related to : Battle of Lake Erie `` Log of the Battle of Lake Erie '' by Sailing Master William
Taylor US Brig Niagara Barclay 's court martial records and correspondence Battle of Lake Erie Historical Sources Animated history of the Battle of Lake Erie ``
Erie , Lake , Battle of '' . The American Cyclopædia. 1879 . Conflicts of the War of 1812 Battles of the War of 1812 United States Washington , D.C. Burning of
Washington Georgia Battle of Fort Peter Louisiana Battle of New Orleans Siege of Fort St. Philip Maryland Battle of Baltimore Battle of Bladensburg Battle of
Caulk 's Field Battle of North Point Battle of St.
</item 7>

<item 8>
Battle of Lake Erie
Fort Bowyer Action of 13 December 1814 Lake Borgne New Orleans Fort St. Philip 2nd Fort Bowyer Pacific Ocean James Island Charles Island Nuku Hiva Downes
Expedition Porter Expedition Typee Valley Valparaiso ( Capture of USS Essex ) Seringapatam Mutiny Action of 9 May 1814 The Battle of Lake Erie , sometimes
called the Battle of Put - in - Bay , was fought on 10 September 1813 , in Lake Erie off the coast of Ohio during the War of 1812 . Nine vessels of the United
States Navy defeated and captured six vessels of the British Royal Navy . This ensured American control of the lake for the rest of the war , which in turn
allowed the Americans to recover Detroit and win the Battle of the Thames to break the Indian confederation of Tecumseh . It was one of the biggest naval
battles of the War of 1812 .
</item 8>

<item 9>
Battle of Lake Erie
Battle of Lake Erie - wikipedia Battle of Lake Erie Jump to : navigation , search Battle of Lake Erie Part of the War of 1812 Battle of Lake Erie by William
Henry Powell , painted 1865 , shows Oliver Hazard Perry transferring from Lawrence to Niagara Date 10 September 1813 Location Lake Erie , near Put - in - Bay ,
Ohio Result Decisive American victory Belligerents United States United Kingdom Commanders and leaders Oliver Hazard Perry Jesse Elliot Robert Heriot Barclay
Robert Finnis Strength 5 schooners 3 brigs 1 sloop 2 ships 1 brig 2 schooners 1 sloop Casualties and losses 27 killed 96 wounded 1 brig severely damaged 41
killed 93 wounded 306 captured entire squadron captured Detroit frontier Tippecanoe Fort Mackinac ( 1812 ) Brownstown Maguaga Fort Dearborn Detroit Fort
Harrison Fort Wayne Wild Cat Creek Mississinewa Frenchtown Africa Point Fort Meigs Fort Stephenson Lake Erie Thames Longwoods Prairie du Chien Campbell Island
Mackinac Island ( 1814 ) Lake Huron Malcolm 's Mills Naval campaigns of the War of 1812 Atlantic Ocean USS Essex vs HMS Alert USS Constitution vs HMS
</item 9>

<item 10>
Battle of Lake Erie
The Americans lost 27 killed and 96 wounded , of whom 2 later died . Of the vessels involved , the three most battered ( the American brig Lawrence and the
British ships Detroit and Queen Charlotte ) were converted into hospital ships . A gale swept the lake on 13 September and dismasted Detroit and Queen Charlotte
, further shattering the already battered ships . Once the wounded had been ferried to Erie , Lawrence was restored to service for 1814 , but the two British
ships were effectively reduced to hulks . Aftermath ( edit ) This section needs additional citations for verification . Please help improve this article by
adding citations to reliable sources . Unsourced material may be challenged and removed . ( September 2017 ) ( Learn how and when to remove this template
message ) Perry 's vessels and prizes were anchored and hasty repairs were underway near West Sister Island when Perry composed his now famous message to
Harrison . Scrawled in pencil on the back of an old envelope , Perry wrote : Dear General : We have met the enemy and they are ours . Two ships , two brigs ,
one schooner and one sloop .
</item 10>
=== END ITEM id=f892d6dc2eb61990 ===

=== ITEM id=2dd2befe0e886d8b ===
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
Sgt . Tom Anderson 2.14 Annie Brawley 2.15 Tom Stagg 2.16 Ned Callan 2.17 James Tyler 2.18 Liz Tyler 2.19 Det . Sgt . James Olson 2.20 Morgan Monroe 2.21 Sean
Healy 2.22 Wallace 2.23 Det . Insp . Rob Breedlove 2.24 Det . Ch . Insp . Garrett Brink 2.25 Kiera Sheridan 2.26 Patrick Spencer 2.27 August Larson 2.28 Sarah
Kay 2.29 Joe O'Donnell 2.30 Alison Walden 2.31 Peter Jensen 2.32 Det . Ch . Insp .
</item 1>

<item 2>
List of The Fall characters
Gibson relies on her heavily during the investigative process . Rose Stagg ( edit ) Rose Stagg ( Valene Kane ) , a woman who had a sexual relationship with Paul
whilst at university . During a night of passion , Paul tried to strangle her to death . As the Belfast Strangler comes to the attention of the public , Stagg
notices the similarities and contacts her close friend Reed Smith . Her husband was unaware the attack had taken place . She is later kidnapped and held captive
by Spector . She is later found unconscious in a car trunk . During series three , Stagg reveals that her sexual relationship with Paul was entirely consensual
, and that , during sex , he strangled her so aggressively she had to be revived . DC Glen Martin ( edit ) DC Glen Martin ( Emmett Scanlan ) , a capable
detective with a somewhat boyish attitude towards his work . Martin is an invaluable member of Gibson 's team and , during series two , takes on a more
prominent role in the decision making process . He is partnered with DC Gail McNally , and is fiercely protective of her , though not to a fault .
</item 2>

<item 3>
List of The Fall characters
Gibson is a very capable detective and a woman who 's highly comfortable with her own sexuality ; this plays a pivotal role in her story arc with other
characters , primarily James Olson and Jim Burns . Whilst working in Belfast , she becomes embroiled in the politics of policing and uncovers corruption within
the force . In the second series , with her investigation stalled , Gibson is forced to get results under the threat of a new 28 - day review into Operation
Musicman . As Stella 's personal past comes back to haunt her , her professional present becomes further tangled in it . Stella has what Paul describes as ``
daddy - issues '' , due to her father 's untimely death . During series three , Gibson is placed under investigation by the Police Ombudsman following the
shooting of Spector . Burns allows her to retain her job , which leads her to fall victim to a vicious assault during her final showdown with the killer . Peter
Paul Spector ( edit ) Peter Paul Spector ( Jamie Dornan ) , the father of two young children .
</item 3>

<item 4>
List of The Fall characters
Katie is a gifted musician and uses online videos to express her feelings . During series 2 she becomes deeply embroiled in Paul 's life . Paul ties her up in a
hotel room , for example , whilst Katie breaks into a house wearing a shirt with Spector 's face on . Benedetto is placed under police surveillance and later
arrested whilst trying to dispose of evidence that implicates Paul in the murders . In series three , she attacks a girl with acid , and is detained for her
crimes . During this time , she reveals to Stella that her father 's death had a severe emotional impact on her , and she has been self - harming . Sally Ann
Spector ( edit ) Sally Ann Spector ( Bronagh Waugh ) , the wife of Paul Spector and a neonatal nurse . She has a deep devotion to Paul , but is wary of his
character as she has never met his family . Sally Ann is the mother of his two children and is unaware of his dark side . Over the course of the series , she
becomes embroiled in his crimes without her knowledge , and even provides a false alibi to the police .
</item 4>

<item 5>
List of The Fall characters
Asst . Ch . Constable Jim Burns ( edit ) ACC Jim Burns ( John Lynch ) , the Assistant Chief Constable of the PSNI in charge of operations , and the Gold
Commander of Musicman . Burns had an affair with Stella several years previously , and , despite being married , still appears to be infatuated with her . He 's
a skilled officer and knows how to balance politics and police - work . He is initially hesitant about linking the murders , but later provides the
investigation with all the resources it requires . He is , at first , believed to be linked to the PSNI corruption scandal , though this is later disproved .
During series 2 , his lust for Stella becomes more apparent when he attempts to force himself on her whilst drunk . This drinking continues into series three ,
and he resigns his commission during the series finale . Katie Benedetto ( edit ) Katerina `` Katie '' Benedetto ( Aisling Franciosi ) , a 15 - year - old
schoolgirl . She babysits the Spectors ' children and - over the course of doing so - develops an affinity for Paul Spector .
</item 5>

<item 6>
List of The Fall characters
Ch . Insp . Joan Kinkead 2.33 David Alvarez 3 References 4 External links Cast overview ( edit ) Actor Character Series 1 Series 2 Series 3 Total Appearances 5
6 7 8 9 10 11 12 13 14 15 16 17 Gillian Anderson DSI Stella Gibson 17 Jamie Dornan Paul Spector 17 John Lynch ACC Jim Burns 17 Aisling Franciosi Katie Benedetto
N / A 16 Bronagh Waugh Sally Ann Spector N / A 15 Niamh McGrady PC Danielle Ferrington N / A N / A 15 Stuart Graham DCI Matthew Eastwood N / A N / A N / A 13
Bronágh Taggart DC Gail McNally N / A 12 Valene Kane Rose Stagg N / A N / A N / A N / A 11 Emmett J. Scanlan DC Glen Martin N / A N / A N / A 10 Archie Panjabi
Prof.
</item 6>

<item 7>
List of The Fall characters
`` BBC Two Sets Premiere Date For ' The Fall ' '' . TVWise . Retrieved 1 May 2013 . External links ( edit ) The Fall at BBC Programmes The Fall on RTÉ The Fall
on Netflix The Fall on IMDb The Fall at Rotten Tomatoes The Fall Trailer at ZDF Enterprises The Fall on Facebook Retrieved from ``
https://en.wikipedia.org/w/index.php?title=List_of_The_Fall_characters&oldid=811099612 '' Categories : Lists of drama television characters Lists of British
television series characters Lists of Irish television series characters Hidden categories : Use dmy dates from January 2017 Use British English from January
2017 BBC programme ID not in Wikidata Talk Contents About Wikipedia Add links This page was last edited on 19 November 2017 , at 14 : 36 . About Wikipedia
</item 7>

<item 8>
List of The Fall characters
List of the Fall characters - wikipedia List of the Fall characters Jump to : navigation , search The Fall , starring Gillian Anderson , is a British - Irish
drama series produced by Artists Studio and created by Allan Cubitt . It premiered in the Republic of Ireland at 21 : 30 on 12 May 2013 , and in the UK on BBC
Two at 21 : 00 on 13 May 2013 . The series features Jamie Dornan , with John Lynch , Bronagh Waugh , Stuart Graham , and Niamh McGrady . Contents ( hide ) 1
Cast overview 2 Main characters 2.1 Det . Supt . Stella Gibson 2.2 Peter Paul Spector 2.3 Asst . Ch . Constable Jim Burns 2.4 Katie Benedetto 2.5 Sally Ann
Spector 2.6 PC Danielle Ferrington 2.7 Det . Ch . Insp . Matthew Eastwood 2.8 DC Gail McNally 2.9 Rose Stagg 2.10 DC Glen Martin 2.11 Professor Reed Smith 2.12
DC Rick Turner 2.13 Det .
</item 8>

<item 9>
Graph query row:
entity: Valene Kane
relationships: ['RELATED_TO', 'RELATED_TO']
related: Rose Stagg
</item 9>

<item 10>
List of The Fall characters
After leaving Paul for a short time , believing he slept with Katie , the two reconcile and she flees Ireland with him . In series 2 , they return . Sally Ann
is pregnant with her third child , though she suffers a miscarriage . Following her arrest for assisting an offender , Sally attempts to kill her children . Pc
Danielle Ferrington ( edit ) PC Danielle Ferrington ( Niamh McGrady ) , a Police Constable first seen attending a breaking and entering call at Sarah Kay 's
home . She blames herself for Kay 's murder and tells Gibson this , resulting in her being recruited by Stella to work with Operation Musicman . Dani is a
skilled investigator , and proves herself invaluable to the team . She speaks openly of her homosexuality to bisexual Gibson , and as a result the two develop
both a professional and a personal bond . She later returns to patrol , believing she can do the most good on the street . It is in this capacity that she is
involved in a gunfight with Spector and , following her suspension from front - line duty , is re-recruited by Stella . Det . Ch . Insp .
</item 10>
=== END ITEM id=2dd2befe0e886d8b ===

=== ITEM id=29c02d313a2b1d9b ===
Gold sentences and facts:
Sentence 0: No. in series No. in season Title Directed by Written by Original air date Production code U.S. viewers ( millions ) 156 `` Playing Dead '' Michael Smith Antoinette Stella April 19 , 2009 ( 2009 - 04 - 19 ) ( USA ) May 6 , 2009 ( 2009 - 05 - 06 ) ( NBC ) 08009 4.57 Detectives Goren and Eames investigate the shooting of Stacy Hayes - Fitzgerald and her boyfriend Rick Siebert , two drug addicts who are blackmailing Stacy 's stepfather , Neil Hayes - Fitzgerald ( Scott Cohen ) , a politician running for mayor . Rick was killed in the shooting , but Stacy survived by playing dead for hours under his bleeding corpse until they were found by Stacy 's mother , Josie . After returning home from the hospital , Stacy attempts to commit suicide . After seeing how she reacts when Neil is around , Goren asks Stacy if she has been abused , but she denies it . Goren discovers that Stacy and her family moved to Switzerland for a year when she was in high school , and realizes that Stacy 's younger sister Sophie is actually her daughter by Neil . While questioning Neil and his mother , Camille ( Kathy Baker ) , it is revealed that she had Rick killed because he was threatening to expose Neil 's family scandal , thwarting his mayoral bid . 157 `` Rock Star '' Bill D'Elia Ed Zuckerman April 26 , 2009 ( 2009 - 04 - 26 ) ( USA ) May 13 , 2009 ( 2009 - 05 - 13 ) ( NBC ) 08008 4.10 Detective Wheeler and her new partner , Detective Zack Nichols , investigate the stabbing of an artist , who was found in a neighborhood rife with ethnic tensions . 158 `` Identity Crisis '' Michael Smith Pamela Wechsler May 3 , 2009 ( 2009 - 05 - 03 ) ( USA ) May 20 , 2009 ( 2009 - 05 - 20 ) ( NBC ) 08005 3.14 Two young brothers live with their schizophrenic mother , who dies in the bathtub by electrocution . They are questioned about the turn of events , and the younger brother Anthony tells police that he saw his brother Tommy push the electric heater into the tub . The two are subsequently separated and put into foster care . Twenty years later , Anthony reappears in Tommy 's life , and Tommy responds by shooting him . Goren and Eames are brought in to investigate the murder , and Rodgers tells them that the victim , Anthony , was dying from lymphoma . Meanwhile , over the years Tommy ( Sam Trammell ) has been living a mysterious life , pretending that he is a graduate of Princeton University and taking on false identities . He rents a house on Nantucket under the name Tyler Chisholm , and the owner 's daughter takes a liking to him . The owner is suspicious and runs a background credit check on `` Tyler . '' The credit check is reported to the detectives , as Tyler Chisholm was one of the names noted when interviewing people at the Princeton Eating Club . They travel to Nantucket , where they find Tommy and take him into custody . Back in New York , Eames questions the owner 's daughter and asks if there was anything strange or out of the ordinary that she could recall . She tells Eames that she remembers Tommy throwing something into the water and staring at it . Goren then questions Tommy and reveals that Anthony did not return to bring back old memories and haunt Tommy , but to make amends because of his imminent death . 159 `` In Treatment '' Jean de Segonzac Timothy J. Lea May 10 , 2009 ( 2009 - 05 - 10 ) ( USA ) May 27 , 2009 ( 2009 - 05 - 27 ) ( NBC ) 08010 3.14 Detectives Nichols and Wheeler investigate the murder of a Wall Street executive who is murdered during a gala . Nichols believes that his company is more involved with the case than they have admitted to the police . During the investigation the detectives discover that news of a falling stock , which plunged more after news of a fraud investigation , is tied to this murder . However , their investigation leads them to a treatment clinic , where their prime suspect is under the care of a controlling psychiatrist ( Dennis Boutsikaris ) . 160 5 `` Faithfully '' Jean de Segonzac Antoinette Stella May 17 , 2009 ( 2009 - 05 - 17 ) ( USA ) June 29 , 2009 ( 2009 - 06 - 29 ) ( NBC ) 08001 3.70 When a devout celebrity doctor is murdered , Goren and Eames enter the unexpectedly steamy world behind the pulpit . 161 6 `` Astoria Helen '' Norberto Barba Timothy J. Lea May 31 , 2009 ( 2009 - 05 - 31 ) ( USA ) July 6 , 2009 ( 2009 - 07 - 06 ) ( NBC ) 08004 3.97 A charming con man named Joe Gallagher ( Creighton James ) targets a lonely , single mother named Helen Bramer ( Arija Bareikis ) to try to hit an armored truck carrying millions of dollars , Bramer worked for the company of the armored truck . Detectives Nichols and Wheeler investigate when one of the con man 's partners dies weeks later in a bomb blast . Bramer and Gallagher become the prime suspects in the case when a van owner named Frank Stroup ( Domenick Lombardozzi ) is ripped off and later shot . But the detectives later learn that Bramer and Gallagher are n't the prime suspects when Bramer 's son Kevin gets abducted by Frank Stroup , and to rescue the boy , Detective Nichols has to put himself in the middle of a hostage situation . 162 7 `` Folie à Deux '' David Manson Michael S. Chernuchin June 7 , 2009 ( 2009 - 06 - 07 ) ( USA ) July 13 , 2009 ( 2009 - 07 - 13 ) ( NBC ) 08003 4.01 Detectives Goren and Eames investigate when a couple 's 2 - year - old daughter mysteriously disappears from her room during a robbery at the hotel , where her parents dined . They learn that a family member ( Lynn Redgrave ) is financially supporting the couple , while also dying of a heart condition . Glass fragments and a stolen key - card become substantial evidence which leads them to the robber , although there is no child with him . Further investigation leads Goren and Eames to receive a ransom call . However , they learn that someone relatively close to the family was taking advantage of the crime , attempting to win money over in the process . The detectives find out the parents have been giving a false description . There never was a child at the hotel . As they uncover a shocking conspiracy , Goren and Eames learn that the unstable mother ( Piper Perabo ) has been delusional about the months - ago accidental death of her daughter , prompting her husband ( Luke Kirby ) to exploit her condition in order to continue receiving financial support from his rich but childless aunt . Partially inspired by the Madeleine McCann and Caylee Anthony cases . 163 8 `` The Glory That Was ... '' Norberto Barba Robert Nathan June 14 , 2009 ( 2009 - 06 - 14 ) ( USA ) July 20 , 2009 ( 2009 - 07 - 20 ) ( NBC ) 08002 4.14 What should have been a simple meeting turns to murder , leaving Detectives Wheeler and Nichols to question an embassy , a security company , the victim 's relatives , and how it all connects to the Olympic Site Selection Committee . This episode is omitted in the DVD release of season 8 . 164 9 `` Family Values '' Jean de Segonzac Teleplay by : Walon Green Story by : Walon Green & Antoinette Stella June 21 , 2009 ( 2009 - 06 - 21 ) ( USA ) August 3 , 2009 ( 2009 - 08 - 03 ) ( NBC ) 08015 3.44 When a flier from a local Catholic high school drama production connects three murders , Detectives Goren and Eames are back on the case to understand the psychology of a religious fanatic before he claims another victim . 165 10 `` Salome in Manhattan '' `` Salome '' Steve Shill Andrew Lipsitz June 28 , 2009 ( 2009 - 06 - 28 ) ( USA ) August 10 , 2009 ( 2009 - 08 - 10 ) ( NBC ) 08006 N / A A celebrity chef 's star is on the rise as he plans the opening of a trendy restaurant with powerful partners . Yet his world is upended when his tabloid - friendly , gorgeous girlfriend and business partner is killed under mysterious circumstances . Detectives Nichols and Wheeler enter the seductive world of power and privilege and discover desperation and murder , but the more they learn about the victim and the men in her life , the more they learn about who the true killer is . 166 11 `` Lady 's Man '' Ken Girotti Michael S. Chernuchin June 28 , 2009 ( 2009 - 06 - 28 ) ( USA ) August 17 , 2009 ( 2009 - 08 - 17 ) ( NBC ) 08011 N / A Detectives Goren and Eames are back on the case to investigate the killing of a man who became famous for dodging a murder conviction because of a mishandled trial . The man spun his notoriety into a TV show but turned up dead on his birthday . Eames had worked the case of his first wife 's murder 10 years earlier . The case also involves an ADA ( Raul Esparza ) who has proven a thorn in Eames ' side more than once . 167 12 `` Passion '' Jonathan Herron Michael S. Chernuchin July 12 , 2009 ( 2009 - 07 - 12 ) ( USA ) August 24 , 2009 ( 2009 - 08 - 24 ) ( NBC ) 08012 3.47 Egocentric poet Jacob Garrety ( Will Chase ) would do anything to keep his poetry journal , `` The Village Quarterly '' , afloat . When foundation president Don McCallum shows up and announces he is cutting funding to the journal , Jacob appeals to assistant and lover Lauren to use her sexuality to change Don 's mind . Though reluctant , she agrees . While Jacob enjoys the advances of arts patron Sandra Dunbar ( Sarah Rafferty ) at a poetry reading , Lauren carries out Jacob 's wish , and winds up dead on the street . Detectives Nichols and Wheeler discover the deadly passion that exists behind the arts . 168 13 `` All In '' David Manson Teleplay by : Pamela Wechsler , Antoinette Stella , & Walon Green Story by : Pamela Wechsler July 19 , 2009 ( 2009 - 07 - 19 ) ( USA ) August 31 , 2009 ( 2009 - 08 - 31 ) ( NBC ) 08013 4.14 When poker genius Josh Snow ( Aaron Stanford ) loses $80,000 in a card game , backer and bookie Lou Cardinale ( Boris McGiver ) forces him into collections to work off the losses . Lou gives him a . 38 and a box of blanks to extract gambling debts , and sends along his mistress , Angela ( Aleksa Palladino ) , to keep an eye on the money . When Josh first collects from a guy in the local neighborhood , he shoots a blank at him . The guy gave him twenty thousand dollars and then he collects from Kip McGonagle ( Robert Leeshock ) . He fires the gun expecting a blank , but a real bullet kills Kip in the street . Detectives Goren and Eames , who know Snow from a former case ( identical historical plot elements , but different actor and character 's name , from season five 's `` Cruise to Nowhere '' ) , enter an elaborate game where they must join in the play , or be played . 169 14 `` Major Case '' Chris Zalla Andrew Lipsitz July 26 , 2009 ( 2009 - 07 - 26 ) ( USA ) August 31 , 2009 ( 2009 - 08 - 31 ) ( NBC ) 08014 4.63 A drug dealer is killed the night before she plans to leave town to live on her cousin 's farm . After Wheeler goes into labor , Eames pairs up with Nichols to investigate and find the girl 's killer . But when Nichols suspects the girl 's killer is an old friend ( Dylan Baker ) who works in the medical examiner 's office , Nichols does all he can to prove that he killed the victim , but it could cost Nichols his job in the process . Last episode to feature Julianne Nicholson ( Detective Megan Wheeler ) ; she is replaced by Kathryn Erbe ( Detective Alexandra Eames ) for the remainder of season eight . As of season nine , she is replaced by Saffron Burrows ( Detective Serena Stevens ) . 170 15 `` Alpha Dog '' Norberto Barba Walon Green August 2 , 2009 ( 2009 - 08 - 02 ) ( USA ) September 7 , 2009 ( 2009 - 09 - 07 ) ( NBC ) 08007 4.23 A hunky posterboy with a million - dollar torso has what others crave : sex appeal . When he shows up dead after a night of sex and drugs with an elusive woman , detectives Goren and Eames uncover the sexual paranoia behind the murder . 171 16 `` Revolution '' John David Coles Michael S. Chernuchin August 9 , 2009 ( 2009 - 08 - 09 ) ( USA ) September 7 , 2009 ( 2009 - 09 - 07 ) ( NBC ) 08016 4.83 A revolutionary plans the kidnapping of a powerful banking executive , but the heist goes wrong and the executive is killed . While probing the murder , Nichols and Eames must also prevent a terrorist campaign . After this episode , Ed Zuckerman and Michael S. Chernuchin leave the show as executive producers . Zuckerman returned to Law & Order as its executive producer and Chernuchin as its co-executive producer . Walon Green takes over as executive producer / show runner for all episodes .
  Fact 0.0: Episode 156, 'Playing Dead', was directed by Michael Smith.
  Fact 0.1: Episode 156, 'Playing Dead', was written by Antoinette Stella.
  Fact 0.2: Episode 156, 'Playing Dead', first aired in the USA on April 19, 2009.
  Fact 0.3: Episode 156, 'Playing Dead', aired on NBC on May 6, 2009.
  Fact 0.4: Episode 156, 'Playing Dead', has production code 08009.
  Fact 0.5: Episode 156, 'Playing Dead', had 4.57 million U.S. viewers.
  Fact 0.6: Detectives Goren and Eames investigate the shooting of Stacy Hayes-Fitzgerald and her boyfriend Rick Siebert.
  Fact 0.7: Stacy Hayes-Fitzgerald and Rick Siebert are two drug addicts who are blackmailing Stacy's stepfather, Neil Hayes-Fitzgerald.
  Fact 0.8: Neil Hayes-Fitzgerald, played by Scott Cohen, is a politician running for mayor.
  Fact 0.9: Rick was killed in the shooting.
  Fact 0.10: Stacy survived the shooting by playing dead for hours under Rick's bleeding corpse.
  Fact 0.11: Stacy was found by her mother, Josie.
  Fact 0.12: After returning home from the hospital, Stacy attempts to commit suicide.
  Fact 0.13: After seeing how Stacy reacts when Neil is around, Goren asks Stacy if she has been abused.
  Fact 0.14: Stacy denies being abused.
  Fact 0.15: Goren discovers that Stacy and her family moved to Switzerland for a year when Stacy was in high school.
  Fact 0.16: Goren realizes that Stacy's younger sister Sophie is actually Stacy's daughter by Neil.
  Fact 0.17: While questioning Neil and his mother, Camille (played by Kathy Baker), it is revealed that Camille had Rick killed.
  Fact 0.18: Camille had Rick killed because Rick was threatening to expose Neil's family scandal, which would thwart Neil's mayoral bid.
  Fact 0.19: Episode 157, 'Rock Star', was directed by Bill D'Elia.
  Fact 0.20: Episode 157, 'Rock Star', was written by Ed Zuckerman.
  Fact 0.21: Episode 157, 'Rock Star', first aired in the USA on April 26, 2009.
  Fact 0.22: Episode 157, 'Rock Star', aired on NBC on May 13, 2009.
  Fact 0.23: Episode 157, 'Rock Star', has production code 08008.
  Fact 0.24: Episode 157, 'Rock Star', had 4.10 million U.S. viewers.
  Fact 0.25: Detective Wheeler and her new partner, Detective Zack Nichols, investigate the stabbing of an artist.
  Fact 0.26: The artist was found in a neighborhood rife with ethnic tensions.
  Fact 0.27: Episode 158, 'Identity Crisis', was directed by Michael Smith.
  Fact 0.28: Episode 158, 'Identity Crisis', was written by Pamela Wechsler.
  Fact 0.29: Episode 158, 'Identity Crisis', first aired in the USA on May 3, 2009.
  Fact 0.30: Episode 158, 'Identity Crisis', aired on NBC on May 20, 2009.
  Fact 0.31: Episode 158, 'Identity Crisis', has production code 08005.
  Fact 0.32: Episode 158, 'Identity Crisis', had 3.14 million U.S. viewers.
  Fact 0.33: Two young brothers live with their schizophrenic mother, who dies in the bathtub by electrocution.
  Fact 0.34: The brothers are questioned about the turn of events.
  Fact 0.35: The younger brother, Anthony, tells police that he saw his brother Tommy push the electric heater into the tub.
  Fact 0.36: The two brothers are separated and put into foster care.
  Fact 0.37: Twenty years later, Anthony reappears in Tommy's life.
  Fact 0.38: Tommy responds by shooting Anthony.
  Fact 0.39: Detectives Goren and Eames are brought in to investigate the murder.
  Fact 0.40: Rodgers tells Goren and Eames that the victim, Anthony, was dying from lymphoma.
  Fact 0.41: Over the years, Tommy (played by Sam Trammell) has been living a mysterious life.
  Fact 0.42: Tommy pretends that he is a graduate of Princeton University and takes on false identities.
  Fact 0.43: Tommy rents a house on Nantucket under the name Tyler Chisholm.
  Fact 0.44: The owner's daughter takes a liking to Tommy.
  Fact 0.45: The owner is suspicious and runs a background credit check on 'Tyler'.
  Fact 0.46: The credit check is reported to the detectives.
  Fact 0.47: Tyler Chisholm was one of the names noted when interviewing people at the Princeton Eating Club.
  Fact 0.48: The detectives travel to Nantucket, where they find Tommy and take him into custody.
  Fact 0.49: Back in New York, Eames questions the owner's daughter and asks if there was anything strange or out of the ordinary that she could recall.
  Fact 0.50: The owner's daughter tells Eames that she remembers Tommy throwing something into the water and staring at it.
  Fact 0.51: Goren questions Tommy and reveals that Anthony did not return to bring back old memories and haunt Tommy.
  Fact 0.52: Goren reveals that Anthony returned to make amends because of Anthony's imminent death.
  Fact 0.53: Episode 159, 'In Treatment', was directed by Jean de Segonzac.
  Fact 0.54: Episode 159, 'In Treatment', was written by Timothy J. Lea.
  Fact 0.55: Episode 159, 'In Treatment', first aired in the USA on May 10, 2009.
  Fact 0.56: Episode 159, 'In Treatment', aired on NBC on May 27, 2009.
  Fact 0.57: Episode 159, 'In Treatment', has production code 08010.
  Fact 0.58: Episode 159, 'In Treatment', had 3.14 million U.S. viewers.
  Fact 0.59: Detectives Nichols and Wheeler investigate the murder of a Wall Street executive who is murdered during a gala.
  Fact 0.60: Nichols believes that the Wall Street executive's company is more involved with the case than the company has admitted to the police.
  Fact 0.61: During the investigation, the detectives discover that news of a falling stock is tied to this murder.
  Fact 0.62: The falling stock plunged more after news of a fraud investigation.
  Fact 0.63: The investigation leads the detectives to a treatment clinic.
  Fact 0.64: Their prime suspect is under the care of a controlling psychiatrist (played by Dennis Boutsikaris).
  Fact 0.65: Episode 160, 'Faithfully', is season 5 episode 5.
  Fact 0.66: Episode 160, 'Faithfully', was directed by Jean de Segonzac.
  Fact 0.67: Episode 160, 'Faithfully', was written by Antoinette Stella.
  Fact 0.68: Episode 160, 'Faithfully', first aired in the USA on May 17, 2009.
  Fact 0.69: Episode 160, 'Faithfully', aired on NBC on June 29, 2009.
  Fact 0.70: Episode 160, 'Faithfully', has production code 08001.
  Fact 0.71: Episode 160, 'Faithfully', had 3.70 million U.S. viewers.
  Fact 0.72: A devout celebrity doctor is murdered.
  Fact 0.73: Goren and Eames enter the unexpectedly steamy world behind the pulpit.
  Fact 0.74: Episode 161, 'Astoria Helen', is season 6 episode 6.
  Fact 0.75: Episode 161, 'Astoria Helen', was directed by Norberto Barba.
  Fact 0.76: Episode 161, 'Astoria Helen', was written by Timothy J. Lea.
  Fact 0.77: Episode 161, 'Astoria Helen', first aired in the USA on May 31, 2009.
  Fact 0.78: Episode 161, 'Astoria Helen', aired on NBC on July 6, 2009.
  Fact 0.79: Episode 161, 'Astoria Helen', has production code 08004.
  Fact 0.80: Episode 161, 'Astoria Helen', had 3.97 million U.S. viewers.
  Fact 0.81: A charming con man named Joe Gallagher (played by Creighton James) targets a lonely, single mother named Helen Bramer (played by Arija Bareikis).
  Fact 0.82: Joe Gallagher targets Helen Bramer to try to hit an armored truck carrying millions of dollars.
  Fact 0.83: Helen Bramer worked for the company of the armored truck.
  Fact 0.84: Detectives Nichols and Wheeler investigate when one of the con man's partners dies weeks later in a bomb blast.
  Fact 0.85: Helen Bramer and Joe Gallagher become the prime suspects when a van owner named Frank Stroup (played by Domenick Lombardozzi) is ripped off and later shot.
  Fact 0.86: The detectives later learn that Helen Bramer and Joe Gallagher are not the prime suspects when Helen Bramer's son Kevin gets abducted by Frank Stroup.
  Fact 0.87: To rescue the boy, Detective Nichols has to put himself in the middle of a hostage situation.
  Fact 0.88: Episode 162, 'Folie a Deux', is season 7 episode 7.
  Fact 0.89: Episode 162, 'Folie a Deux', was directed by David Manson.
  Fact 0.90: Episode 162, 'Folie a Deux', was written by Michael S. Chernuchin.
  Fact 0.91: Episode 162, 'Folie a Deux', first aired in the USA on June 7, 2009.
  Fact 0.92: Episode 162, 'Folie a Deux', aired on NBC on July 13, 2009.
  Fact 0.93: Episode 162, 'Folie a Deux', has production code 08003.
  Fact 0.94: Episode 162, 'Folie a Deux', had 4.01 million U.S. viewers.
  Fact 0.95: Detectives Goren and Eames investigate when a couple's 2-year-old daughter mysteriously disappears from her room during a robbery at the hotel where her parents dined.
  Fact 0.96: The couple is financially supported by a family member (played by Lynn Redgrave), who is also dying of a heart condition.
  Fact 0.97: Glass fragments and a stolen key-card become substantial evidence that leads the detectives to the robber, although there is no child with him.
  Fact 0.98: Goren and Eames receive a ransom call.
  Fact 0.99: They learn that someone relatively close to the family was taking advantage of the crime, attempting to win money in the process.
  Fact 0.100: The detectives find out the parents have been giving a false description.
  Fact 0.101: There never was a child at the hotel.
  Fact 0.102: The unstable mother (played by Piper Perabo) has been delusional about the months-ago accidental death of her daughter.
  Fact 0.103: The husband (played by Luke Kirby) exploits his wife's condition to continue receiving financial support from his rich but childless aunt.
  Fact 0.104: The episode is partially inspired by the Madeleine McCann and Caylee Anthony cases.
  Fact 0.105: Episode 163, 'The Glory That Was ...', is season 8 episode 8.
  Fact 0.106: Episode 163, 'The Glory That Was ...', was directed by Norberto Barba.
  Fact 0.107: Episode 163, 'The Glory That Was ...', was written by Robert Nathan.
  Fact 0.108: Episode 163, 'The Glory That Was ...', first aired in the USA on June 14, 2009.
  Fact 0.109: Episode 163, 'The Glory That Was ...', aired on NBC on July 20, 2009.
  Fact 0.110: Episode 163, 'The Glory That Was ...', has production code 08002.
  Fact 0.111: Episode 163, 'The Glory That Was ...', had 4.14 million U.S. viewers.
  Fact 0.112: What should have been a simple meeting turns to murder.
  Fact 0.113: Detectives Wheeler and Nichols question an embassy, a security company, the victim's relatives, and how it all connects to the Olympic Site Selection Committee.
  Fact 0.114: Episode 163, 'The Glory That Was ...', is omitted in the DVD release of season 8.
  Fact 0.115: Episode 164, 'Family Values', is season 9 episode 9.
  Fact 0.116: Episode 164, 'Family Values', was directed by Jean de Segonzac.
  Fact 0.117: Episode 164, 'Family Values', has a teleplay by Walon Green.
  Fact 0.118: Episode 164, 'Family Values', has a story by Walon Green and Antoinette Stella.
  Fact 0.119: Episode 164, 'Family Values', first aired in the USA on June 21, 2009.
  Fact 0.120: Episode 164, 'Family Values', aired on NBC on August 3, 2009.
  Fact 0.121: Episode 164, 'Family Values', has production code 08015.
  Fact 0.122: Episode 164, 'Family Values', had 3.44 million U.S. viewers.
  Fact 0.123: When a flier from a local Catholic high school drama production connects three murders, Detectives Goren and Eames are back on the case.
  Fact 0.124: Goren and Eames want to understand the psychology of a religious fanatic before he claims another victim.
  Fact 0.125: Episode 165, 'Salome in Manhattan', is season 10 episode 10.
  Fact 0.126: Episode 165 is also listed under the title 'Salome'.
  Fact 0.127: Episode 165, 'Salome in Manhattan', was directed by Steve Shill.
  Fact 0.128: Episode 165, 'Salome in Manhattan', was written by Andrew Lipsitz.
  Fact 0.129: Episode 165, 'Salome in Manhattan', first aired in the USA on June 28, 2009.
  Fact 0.130: Episode 165, 'Salome in Manhattan', aired on NBC on August 10, 2009.
  Fact 0.131: Episode 165, 'Salome in Manhattan', has production code 08006.
  Fact 0.132: Episode 165, 'Salome in Manhattan', has no U.S. viewer figure (N/A).
  Fact 0.133: A celebrity chef's star is on the rise as he plans the opening of a trendy restaurant with powerful partners.
  Fact 0.134: The chef's tabloid-friendly, gorgeous girlfriend and business partner is killed under mysterious circumstances.
  Fact 0.135: Detectives Nichols and Wheeler enter the seductive world of power and privilege.
  Fact 0.136: Detectives Nichols and Wheeler discover desperation and murder.
  Fact 0.137: The more the detectives learn about the victim and the men in her life, the more they learn about who the true killer is.
  Fact 0.138: Episode 166, 'Lady's Man', is season 11 episode 11.
  Fact 0.139: Episode 166, 'Lady's Man', was directed by Ken Girotti.
  Fact 0.140: Episode 166, 'Lady's Man', was written by Michael S. Chernuchin.
  Fact 0.141: Episode 166, 'Lady's Man', first aired in the USA on June 28, 2009.
  Fact 0.142: Episode 166, 'Lady's Man', aired on NBC on August 17, 2009.
  Fact 0.143: Episode 166, 'Lady's Man', has production code 08011.
  Fact 0.144: Episode 166, 'Lady's Man', has no U.S. viewer figure (N/A).
  Fact 0.145: Detectives Goren and Eames investigate the killing of a man who became famous for dodging a murder conviction because of a mishandled trial.
  Fact 0.146: The man spun his notoriety into a TV show but turned up dead on his birthday.
  Fact 0.147: Eames had worked the case of his first wife's murder 10 years earlier.
  Fact 0.148: The case also involves an ADA (played by Raul Esparza) who has proven a thorn in Eames' side more than once.
  Fact 0.149: Episode 167, 'Passion', is season 12 episode 12.
  Fact 0.150: Episode 167, 'Passion', was directed by Jonathan Herron.
  Fact 0.151: Episode 167, 'Passion', was written by Michael S. Chernuchin.
  Fact 0.152: Episode 167, 'Passion', first aired in the USA on July 12, 2009.
  Fact 0.153: Episode 167, 'Passion', aired on NBC on August 24, 2009.
  Fact 0.154: Episode 167, 'Passion', has production code 08012.
  Fact 0.155: Episode 167, 'Passion', had 3.47 million U.S. viewers.
  Fact 0.156: Egocentric poet Jacob Garrety (played by Will Chase) would do anything to keep his poetry journal, 'The Village Quarterly', afloat.
  Fact 0.157: Foundation president Don McCallum announces he is cutting funding to the journal.
  Fact 0.158: Jacob appeals to his assistant and lover Lauren to use her sexuality to change Don McCallum's mind.
  Fact 0.159: Lauren reluctantly agrees to use her sexuality to change Don McCallum's mind.
  Fact 0.160: While Jacob enjoys the advances of arts patron Sandra Dunbar (played by Sarah Rafferty) at a poetry reading, Lauren carries out Jacob's wish.
  Fact 0.161: Lauren winds up dead on the street.
  Fact 0.162: Detectives Nichols and Wheeler discover the deadly passion that exists behind the arts.
  Fact 0.163: Episode 168, 'All In', is season 13 episode 13.
  Fact 0.164: Episode 168, 'All In', was directed by David Manson.
  Fact 0.165: Episode 168, 'All In', has a teleplay by Pamela Wechsler, Antoinette Stella, and Walon Green.
  Fact 0.166: Episode 168, 'All In', has a story by Pamela Wechsler.
  Fact 0.167: Episode 168, 'All In', first aired in the USA on July 19, 2009.
  Fact 0.168: Episode 168, 'All In', aired on NBC on August 31, 2009.
  Fact 0.169: Episode 168, 'All In', has production code 08013.
  Fact 0.170: Episode 168, 'All In', had 4.14 million U.S. viewers.
  Fact 0.171: When poker genius Josh Snow (played by Aaron Stanford) loses $80,000 in a card game, backer and bookie Lou Cardinale (played by Boris McGiver) forces Josh Snow into collections to work off the losses.
  Fact 0.172: Lou Cardinale gives Josh Snow a .38 and a box of blanks to extract gambling debts.
  Fact 0.173: Lou Cardinale sends along his mistress, Angela (played by Aleksa Palladino), to keep an eye on the money.
  Fact 0.174: When Josh Snow first collects from a guy in the local neighborhood, he shoots a blank at him.
  Fact 0.175: The guy gave Josh Snow twenty thousand dollars.
  Fact 0.176: Josh Snow then collects from Kip McGonagle (played by Robert Leeshock).
  Fact 0.177: Josh Snow fires the gun expecting a blank, but a real bullet kills Kip McGonagle in the street.
  Fact 0.178: Detectives Goren and Eames know Josh Snow from a former case.
  Fact 0.179: The former case is identical in historical plot elements to this episode but has a different actor and character name, from season five's 'Cruise to Nowhere'.
  Fact 0.180: Goren and Eames enter an elaborate game where they must join in the play, or be played.
  Fact 0.181: Episode 169, 'Major Case', is season 14 episode 14.
  Fact 0.182: Episode 169, 'Major Case', was directed by Chris Zalla.
  Fact 0.183: Episode 169, 'Major Case', was written by Andrew Lipsitz.
  Fact 0.184: Episode 169, 'Major Case', first aired in the USA on July 26, 2009.
  Fact 0.185: Episode 169, 'Major Case', aired on NBC on August 31, 2009.
  Fact 0.186: Episode 169, 'Major Case', has production code 08014.
  Fact 0.187: Episode 169, 'Major Case', had 4.63 million U.S. viewers.
  Fact 0.188: A drug dealer is killed the night before she plans to leave town to live on her cousin's farm.
  Fact 0.189: After Detective Megan Wheeler goes into labor, Detective Eames pairs up with Detective Nichols to investigate and find the girl's killer.
  Fact 0.190: Nichols suspects the girl's killer is an old friend (played by Dylan Baker) who works in the medical examiner's office.
  Fact 0.191: Nichols does all he can to prove that the old friend killed the victim, but it could cost Nichols his job in the process.
  Fact 0.192: Episode 169, 'Major Case', is the last episode to feature Julianne Nicholson as Detective Megan Wheeler.
  Fact 0.193: Detective Megan Wheeler is replaced by Kathryn Erbe (Detective Alexandra Eames) for the remainder of season eight.
  Fact 0.194: As of season nine, Detective Megan Wheeler is replaced by Saffron Burrows (Detective Serena Stevens).
  Fact 0.195: Episode 170, 'Alpha Dog', is season 15 episode 15.
  Fact 0.196: Episode 170, 'Alpha Dog', was directed by Norberto Barba.
  Fact 0.197: Episode 170, 'Alpha Dog', was written by Walon Green.
  Fact 0.198: Episode 170, 'Alpha Dog', first aired in the USA on August 2, 2009.
  Fact 0.199: Episode 170, 'Alpha Dog', aired on NBC on September 7, 2009.
  Fact 0.200: Episode 170, 'Alpha Dog', has production code 08007.
  Fact 0.201: Episode 170, 'Alpha Dog', had 4.23 million U.S. viewers.
  Fact 0.202: A hunky posterboy with a million-dollar torso has what others crave: sex appeal.
  Fact 0.203: When the posterboy shows up dead after a night of sex and drugs with an elusive woman, detectives Goren and Eames uncover the sexual paranoia behind the murder.
  Fact 0.204: Episode 171, 'Revolution', is season 16 episode 16.
  Fact 0.205: Episode 171, 'Revolution', was directed by John David Coles.
  Fact 0.206: Episode 171, 'Revolution', was written by Michael S. Chernuchin.
  Fact 0.207: Episode 171, 'Revolution', first aired in the USA on August 9, 2009.
  Fact 0.208: Episode 171, 'Revolution', aired on NBC on September 7, 2009.
  Fact 0.209: Episode 171, 'Revolution', has production code 08016.
  Fact 0.210: Episode 171, 'Revolution', had 4.83 million U.S. viewers.
  Fact 0.211: A revolutionary plans the kidnapping of a powerful banking executive.
  Fact 0.212: The heist goes wrong and the banking executive is killed.
  Fact 0.213: While probing the murder, Nichols and Eames must also prevent a terrorist campaign.
  Fact 0.214: After Episode 171, 'Revolution', Ed Zuckerman and Michael S. Chernuchin leave the show as executive producers.
  Fact 0.215: Ed Zuckerman returned to Law & Order as its executive producer.
  Fact 0.216: Michael S. Chernuchin returned to Law & Order as its co-executive producer.
  Fact 0.217: Walon Green takes over as executive producer and show runner for all episodes.

Retrieved items:
<item 1>
Law & Order: Criminal Intent (season 8)
Detectives Nichols and Wheeler discover the deadly passion that exists behind the arts . 168 13 `` All In '' David Manson Teleplay by : Pamela Wechsler ,
Antoinette Stella , & Walon Green Story by : Pamela Wechsler July 19 , 2009 ( 2009 - 07 - 19 ) ( USA ) August 31 , 2009 ( 2009 - 08 - 31 ) ( NBC ) 08013 4.14
When poker genius Josh Snow ( Aaron Stanford ) loses $80,000 in a card game , backer and bookie Lou Cardinale ( Boris McGiver ) forces him into collections to
work off the losses . Lou gives him a . 38 and a box of blanks to extract gambling debts , and sends along his mistress , Angela ( Aleksa Palladino ) , to keep
an eye on the money . When Josh first collects from a guy in the local neighborhood , he shoots a blank at him . The guy gave him twenty thousand dollars and
then he collects from Kip McGonagle ( Robert Leeshock ) .
</item 1>

<item 2>
Law & Order: Criminal Intent (season 8)
There never was a child at the hotel . As they uncover a shocking conspiracy , Goren and Eames learn that the unstable mother ( Piper Perabo ) has been
delusional about the months - ago accidental death of her daughter , prompting her husband ( Luke Kirby ) to exploit her condition in order to continue
receiving financial support from his rich but childless aunt . Partially inspired by the Madeleine McCann and Caylee Anthony cases . 163 8 `` The Glory That Was
... '' Norberto Barba Robert Nathan June 14 , 2009 ( 2009 - 06 - 14 ) ( USA ) July 20 , 2009 ( 2009 - 07 - 20 ) ( NBC ) 08002 4.14 What should have been a
simple meeting turns to murder , leaving Detectives Wheeler and Nichols to question an embassy , a security company , the victim 's relatives , and how it all
connects to the Olympic Site Selection Committee . This episode is omitted in the DVD release of season 8 .
</item 2>

<item 3>
Law & Order: Criminal Intent (season 8)
This episode is omitted in the DVD release of season 8 . 164 9 `` Family Values '' Jean de Segonzac Teleplay by : Walon Green Story by : Walon Green &
Antoinette Stella June 21 , 2009 ( 2009 - 06 - 21 ) ( USA ) August 3 , 2009 ( 2009 - 08 - 03 ) ( NBC ) 08015 3.44 When a flier from a local Catholic high school
drama production connects three murders , Detectives Goren and Eames are back on the case to understand the psychology of a religious fanatic before he claims
another victim . 165 10 `` Salome in Manhattan '' `` Salome '' Steve Shill Andrew Lipsitz June 28 , 2009 ( 2009 - 06 - 28 ) ( USA ) August 10 , 2009 ( 2009 - 08
- 10 ) ( NBC ) 08006 N / A A celebrity chef 's star is on the rise as he plans the opening of a trendy restaurant with powerful partners .
</item 3>

<item 4>
Law & Order: Criminal Intent (season 8)
Nichols believes that his company is more involved with the case than they have admitted to the police . During the investigation the detectives discover that
news of a falling stock , which plunged more after news of a fraud investigation , is tied to this murder . However , their investigation leads them to a
treatment clinic , where their prime suspect is under the care of a controlling psychiatrist ( Dennis Boutsikaris ) . 160 5 `` Faithfully '' Jean de Segonzac
Antoinette Stella May 17 , 2009 ( 2009 - 05 - 17 ) ( USA ) June 29 , 2009 ( 2009 - 06 - 29 ) ( NBC ) 08001 3.70 When a devout celebrity doctor is murdered ,
Goren and Eames enter the unexpectedly steamy world behind the pulpit .
</item 4>

<item 5>
Law & Order: Criminal Intent (season 8)
161 6 `` Astoria Helen '' Norberto Barba Timothy J. Lea May 31 , 2009 ( 2009 - 05 - 31 ) ( USA ) July 6 , 2009 ( 2009 - 07 - 06 ) ( NBC ) 08004 3.97 A charming
con man named Joe Gallagher ( Creighton James ) targets a lonely , single mother named Helen Bramer ( Arija Bareikis ) to try to hit an armored truck carrying
millions of dollars , Bramer worked for the company of the armored truck . Detectives Nichols and Wheeler investigate when one of the con man 's partners dies
weeks later in a bomb blast . Bramer and Gallagher become the prime suspects in the case when a van owner named Frank Stroup ( Domenick Lombardozzi ) is ripped
off and later shot . But the detectives later learn that Bramer and Gallagher are n't the prime suspects when Bramer 's son Kevin gets abducted by Frank Stroup
, and to rescue the boy , Detective Nichols has to put himself in the middle of a hostage situation .
</item 5>

<item 6>
Law & Order: Criminal Intent (season 8)
Yet his world is upended when his tabloid - friendly , gorgeous girlfriend and business partner is killed under mysterious circumstances . Detectives Nichols
and Wheeler enter the seductive world of power and privilege and discover desperation and murder , but the more they learn about the victim and the men in her
life , the more they learn about who the true killer is . 166 11 `` Lady 's Man '' Ken Girotti Michael S. Chernuchin June 28 , 2009 ( 2009 - 06 - 28 ) ( USA )
August 17 , 2009 ( 2009 - 08 - 17 ) ( NBC ) 08011 N / A Detectives Goren and Eames are back on the case to investigate the killing of a man who became famous
for dodging a murder conviction because of a mishandled trial . The man spun his notoriety into a TV show but turned up dead on his birthday . Eames had worked
the case of his first wife 's murder 10 years earlier .
</item 6>

<item 7>
Law & Order: Criminal Intent (season 8)
.. '' `` Family Values '' `` Major Case '' `` Revolution '' `` Loyalty '' `` To the Boy in the Blue Knit Cap '' See also Episodes Cast Video game Law & Order
franchise Retrieved from `` https://en.wikipedia.org/w/index.php?title=Law_%26_Order:_Criminal_Intent_(season_8)&oldid=815988037 '' Categories : Law & Order :
Criminal Intent episodes 2009 American television seasons Hidden categories : Pages using citations with accessdate and no URL CS1 maint : Unfit url All
articles with dead external links Articles with dead external links from December 2017 Articles with permanently dead external links Pages using deprecated
image syntax Infobox television season articles that use the season name parameter Talk Contents About Wikipedia Wikiquote Français Italiano Nederlands
Srpskohrvatski / српскохрватски Edit links This page was last edited on 18 December 2017 , at 14 : 29 . Text is available under the Creative Commons Attribution
- ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and Privacy Policy .
</item 7>

<item 8>
Law & Order: Criminal Intent (season 8)
Law & Order : Criminal Intent ( season 8 ) - wikipedia Law & Order : Criminal Intent ( season 8 ) Jump to : navigation , search Law & Order : Criminal Intent (
season 8 ) Season 8 U.S. DVD cover Country of origin United States No. of episodes 16 Release Original network USA Network Original release April 19 ( 2009 - 04
- 19 ) -- August 9 , 2009 ( 2009 - 08 - 09 ) Season chronology ← Previous Season 7 Next → Season 9 List of Law & Order : Criminal Intent episodes The eighth
season of Law & Order : Criminal Intent premiered on the USA Network in the United States on April 19 , 2009 . It consisted of sixteen episodes , and concluded
on August 9 , 2009 . The day following each episode 's broadcast on television , they are made available to purchase and download from the iTunes Store . Law &
Order : Criminal Intent is an American police procedural television series set and filmed in New York City .
</item 8>

<item 9>
Law & Order: Criminal Intent (season 8)
162 7 `` Folie à Deux '' David Manson Michael S. Chernuchin June 7 , 2009 ( 2009 - 06 - 07 ) ( USA ) July 13 , 2009 ( 2009 - 07 - 13 ) ( NBC ) 08003 4.01
Detectives Goren and Eames investigate when a couple 's 2 - year - old daughter mysteriously disappears from her room during a robbery at the hotel , where her
parents dined . They learn that a family member ( Lynn Redgrave ) is financially supporting the couple , while also dying of a heart condition . Glass fragments
and a stolen key - card become substantial evidence which leads them to the robber , although there is no child with him . Further investigation leads Goren and
Eames to receive a ransom call . However , they learn that someone relatively close to the family was taking advantage of the crime , attempting to win money
over in the process . The detectives find out the parents have been giving a false description . There never was a child at the hotel .
</item 9>

<item 10>
Law & Order: Criminal Intent (season 8)
in season Title Directed by Written by Original air date Production code U.S. viewers ( millions ) 156 `` Playing Dead '' Michael Smith Antoinette Stella April
19 , 2009 ( 2009 - 04 - 19 ) ( USA ) May 6 , 2009 ( 2009 - 05 - 06 ) ( NBC ) 08009 4.57 Detectives Goren and Eames investigate the shooting of Stacy Hayes -
Fitzgerald and her boyfriend Rick Siebert , two drug addicts who are blackmailing Stacy 's stepfather , Neil Hayes - Fitzgerald ( Scott Cohen ) , a politician
running for mayor . Rick was killed in the shooting , but Stacy survived by playing dead for hours under his bleeding corpse until they were found by Stacy 's
mother , Josie . After returning home from the hospital , Stacy attempts to commit suicide . After seeing how she reacts when Neil is around , Goren asks Stacy
if she has been abused , but she denies it .
</item 10>
=== END ITEM id=29c02d313a2b1d9b ===

=== ITEM id=0d7c74b46f1ed3d1 ===
Gold sentences and facts:
Sentence 0: The Director of the United States Mint is the head of the United States Mint . The position is currently vacant . Since January 20 , 2017 , the senior career official at the Mint has been Acting Principal Deputy Director David Motl .
  Fact 0.0: The Director of the United States Mint is the head of the United States Mint.
  Fact 0.1: The position of Director of the United States Mint is currently vacant.
  Fact 0.2: Since January 20, 2017, the senior career official at the Mint has been David Motl.
  Fact 0.3: David Motl is the Acting Principal Deputy Director.
Sentence 1: Name Assumed Office Left Office President Appointed By David Rittenhouse April 1792 June 1795 George Washington Henry William de Saussure July 1795 October 1795 Elias Boudinot October 1795 July 1805 Robert Patterson January 1806 July 1824 Thomas Jefferson Samuel Moore July 1824 July 1835 James Monroe Robert M. Patterson May 1835 July 1851 Andrew Jackson George Nicholas Eckert July 1851 April 1853 Millard Fillmore Thomas M. Pettit April 1853 May 1853 Franklin Pierce James Ross Snowden June 1853 May 1861 James Pollock May 1861 September 1866 Abraham Lincoln William Millward October 1866 April 1867 Andrew Johnson Henry Linderman April 1867 May 1869 James Pollock May 1869 March 1873 Ulysses S. Grant Henry Linderman April 1873 December 1878 Horatio C. Burchard February 1879 June 1885 Rutherford B. Hayes James P. Kimball July 1885 October 1889 Grover Cleveland Edward O. Leech October 1889 May 1893 Benjamin Harrison Robert E. Preston November 1893 February 1898 Grover Cleveland George E. Roberts February 1898 July 1907 William McKinley Frank A. Leach September 1907 August 1909 Theodore Roosevelt A. Piatt Andrew November 1909 June 1910 William Howard Taft George E. Roberts July 1910 November 1914 Robert W. Woolley March 1915 July 1916 Woodrow Wilson F.H. von Engelken September 1916 March 1917 Raymond T. Baker March 1917 March 1922 Frank Edgar Scobey March 1922 September 1923 Warren G. Harding Robert J. Grant November 1923 May 1933 Calvin Coolidge Nellie Tayloe Ross May 1933 April 1953 Franklin D. Roosevelt William H. Brett July 1954 January 1961 Dwight D. Eisenhower Eva Adams October 1961 August 1969 John F. Kennedy Mary Brooks September 1969 February 1977 Richard Nixon Stella Hackel Sims November 1977 April 1981 Jimmy Carter Donna Pope July 1981 August 1991 Ronald Reagan David J. Ryder September 1992 November 1993 George H.W. Bush Philip N. Diehl June 1994 March 2000 Bill Clinton Jay W. Johnson May 2000 August 2001 Henrietta H. Fore August 2001 August 2005 George W. Bush Edmund C. Moy September 2006 January 2011 David J. Ryder April 2018 Incumbent Donald Trump
  Fact 1.0: David Rittenhouse held the office of Director of the Mint from April 1792 to June 1795, appointed by George Washington.
  Fact 1.1: Henry William de Saussure held the office from July 1795 to October 1795.
  Fact 1.2: Elias Boudinot held the office from October 1795 to July 1805.
  Fact 1.3: Robert Patterson held the office from January 1806 to July 1824, appointed by Thomas Jefferson.
  Fact 1.4: Samuel Moore held the office from July 1824 to July 1835, appointed by James Monroe.
  Fact 1.5: Robert M. Patterson held the office from May 1835 to July 1851, appointed by Andrew Jackson.
  Fact 1.6: George Nicholas Eckert held the office from July 1851 to April 1853, appointed by Millard Fillmore.
  Fact 1.7: Thomas M. Pettit held the office from April 1853 to May 1853, appointed by Franklin Pierce.
  Fact 1.8: James Ross Snowden held the office from June 1853 to May 1861.
  Fact 1.9: James Pollock held the office from May 1861 to September 1866, appointed by Abraham Lincoln.
  Fact 1.10: William Millward held the office from October 1866 to April 1867, appointed by Andrew Johnson.
  Fact 1.11: Henry Linderman held the office from April 1867 to May 1869.
  Fact 1.12: James Pollock held the office from May 1869 to March 1873, appointed by Ulysses S. Grant.
  Fact 1.13: Henry Linderman held the office from April 1873 to December 1878.
  Fact 1.14: Horatio C. Burchard held the office from February 1879 to June 1885, appointed by Rutherford B. Hayes.
  Fact 1.15: James P. Kimball held the office from July 1885 to October 1889, appointed by Grover Cleveland.
  Fact 1.16: Edward O. Leech held the office from October 1889 to May 1893, appointed by Benjamin Harrison.
  Fact 1.17: Robert E. Preston held the office from November 1893 to February 1898, appointed by Grover Cleveland.
  Fact 1.18: George E. Roberts held the office from February 1898 to July 1907, appointed by William McKinley.
  Fact 1.19: Frank A. Leach held the office from September 1907 to August 1909, appointed by Theodore Roosevelt.
  Fact 1.20: A. Piatt Andrew held the office from November 1909 to June 1910.
  Fact 1.21: George E. Roberts held the office from July 1910 to November 1914, appointed by William Howard Taft.
  Fact 1.22: Robert W. Woolley held the office from March 1915 to July 1916, appointed by Woodrow Wilson.
  Fact 1.23: F.H. von Engelken held the office from September 1916 to March 1917.
  Fact 1.24: Raymond T. Baker held the office from March 1917 to March 1922.
  Fact 1.25: Frank Edgar Scobey held the office from March 1922 to September 1923, appointed by Warren G. Harding.
  Fact 1.26: Robert J. Grant held the office from November 1923 to May 1933, appointed by Calvin Coolidge.
  Fact 1.27: Nellie Tayloe Ross held the office from May 1933 to April 1953, appointed by Franklin D. Roosevelt.
  Fact 1.28: William H. Brett held the office from July 1954 to January 1961, appointed by Dwight D. Eisenhower.
  Fact 1.29: Eva Adams held the office from October 1961 to August 1969, appointed by John F. Kennedy.
  Fact 1.30: Mary Brooks held the office from September 1969 to February 1977, appointed by Richard Nixon.
  Fact 1.31: Stella Hackel Sims held the office from November 1977 to April 1981, appointed by Jimmy Carter.
  Fact 1.32: Donna Pope held the office from July 1981 to August 1991, appointed by Ronald Reagan.
  Fact 1.33: David J. Ryder held the office from September 1992 to November 1993, appointed by George H.W. Bush.
  Fact 1.34: Philip N. Diehl held the office from June 1994 to March 2000, appointed by Bill Clinton.
  Fact 1.35: Jay W. Johnson held the office from May 2000 to August 2001.
  Fact 1.36: Henrietta H. Fore held the office from August 2001 to August 2005, appointed by George W. Bush.
  Fact 1.37: Edmund C. Moy held the office from September 2006 to January 2011.
  Fact 1.38: David J. Ryder has held the office since April 2018, appointed by Donald Trump.

Retrieved items:
<item 1>
Vice President of the United States
Senate vice presidential bust collection In fiction Presidents actors Vice Presidents actors Candidates Line of succession U.S. Government Portal Biography
Portal Presidents of the United States Presidential line of succession in the United States of America Vice President ( Mike Pence ) Speaker of the House of
Representatives ( Paul Ryan ) President pro tempore of the Senate ( Orrin Hatch ) Secretary of State ( Mike Pompeo ) Secretary of the Treasury ( Steven Mnuchin
) Secretary of Defense ( Jim Mattis ) Attorney General ( Jeff Sessions ) Secretary of the Interior ( Ryan Zinke ) Secretary of Agriculture ( Sonny Perdue )
Secretary of Commerce ( Wilbur Ross ) Secretary of Labor ( Alex Acosta ) Secretary of Health and Human Services ( Alex Azar ) Secretary of Housing and Urban
Development ( Ben Carson ) Secretary of Transportation ( Elaine Chao ) Secretary of Energy ( Rick Perry ) Secretary of Education ( Betsy DeVos ) Secretary of
Veterans Affairs ( Robert Wilkie ) Secretary of Homeland Security ( Kirstjen Nielsen ) Jump up ^ Although Elaine Chao is the current Secretary of Transportation
, she is not a natural - born citizen ( acquired U.S. citizenship by naturalization ) and thus ineligible for the presidency .
</item 1>

<item 2>
Director of the United States Mint
Director of the United States Mint - wikipedia Director of the United States Mint Jump to : navigation , search Seal of the United States Mint . The Director of
the United States Mint is the head of the United States Mint . The position is currently vacant . Since January 20 , 2017 , the senior career official at the
Mint has been Acting Principal Deputy Director David Motl . The office of Director has existed since the creation of the Mint by the Coinage Act of 1792 .
Initially appointed serving at the pleasure of the President of the United States , the Coinage Act of 1873 specified a five - year term for Directors . The
Director operates with general directions provided by the United States Secretary of the Treasury . Since the resignation of Edmund C. Moy in 2011 the mint has
been without an official director . Initially Matthew Rhett Jeppson was nominated by US President Barack Obama to become the mint 's 39th director and was given
the temporary title of principal deputy director , however the nomination was never confirmed by the Senate . Jeppson stepped down as principal deputy director
in January 2017 , being replaced by Acting Principal Deputy Director David Motl .
</item 2>

<item 3>
United States two-dollar bill
National Bank Notes were issued in 1875 and feature a woman unfurling a flag and a big 2 ( Lazy Duce ) on the obverse , the reverse has the king of England
smoking tobacco and an eagle with a shield . The second two - dollar denomination in the silver certificate series printed in 1891 . This note features United
States Secretary of the Treasury William Windom . In 1886 , the first $2 silver certificate with a portrait of United States Civil War General Winfield Scott
Hancock on the left of the obverse was issued . This design went on until 1891 when a new $2 Silver Certificate was issued with a portrait of U.S. Treasury
Secretary William Windom in the center of the obverse . Two - dollar Treasury , or `` Coin '' , Notes were first issued for government purchases of silver
bullion in 1890 from the silver mining industry . The reverse featured large wording of TWO in the center and a numeral 2 to the right surrounded by an ornate
design that occupied almost the entire note .
</item 3>

<item 4>
Large denominations of United States currency
dollar U.S. Mint Philadelphia San Francisco Denver West Point Bureau of Engraving and Printing Mutilated currency Current coinage Penny ( 1 ¢ ) Nickel ( 5 ¢ )
Dime ( 10 ¢ ) Quarter ( 25 ¢ ) Half dollar ( 50 ¢ ) Dollar ( $1 ) Bullion coinage America the Beautiful Silver Bullion Coins American Buffalo American Gold
Eagle American Platinum Eagle American Silver Eagle American Palladium Eagle Current paper money $1 $2 $5 $10 $20 $50 $100 See also Bicentennial coinage
Commemoratives Early Commemoratives Modern Commemoratives Confederate dollar Large denominations Obsolete denominations Promotional fake denominations
Counterfeit United States currency Mill Coin production In God We Trust E pluribus unum Annuit cœptis Nicknames Replacement banknote Sales tax token ( hide )
Obsolete United States currency and coinage Topics United States coinage United States dollar History of the United States dollar Coins Half cent ( 1793 -- 1857
) Fugio Cent ( 1787 ) Large cent ( 1793 -- 1857 ;
</item 4>

<item 5>
Director of the United States Mint
Bush Philip N. Diehl June 1994 March 2000 Bill Clinton Jay W. Johnson May 2000 August 2001 Henrietta H. Fore August 2001 August 2005 George W. Bush Edmund C.
Moy September 2006 January 2011 David J. Ryder April 2018 Incumbent Donald Trump References ( edit ) Jump up ^ About the Mint at usmint.gov ( retrieved 20 April
2015 ) . Jump up ^ Gilkes , Paul ( 23 January 2017 ) . `` Rhett Jeppson leaves U.S. Mint as principal deputy director '' . coinworld.com . Retrieved 13 April
2017 . Jump up ^ `` Director 's Office '' . US Mint . Retrieved 13 April 2017 .
</item 5>

<item 6>
United States two-dollar bill
dollar U.S. Mint Philadelphia San Francisco Denver West Point Bureau of Engraving and Printing Mutilated currency Current coinage Penny ( 1 ¢ ) Nickel ( 5 ¢ )
Dime ( 10 ¢ ) Quarter ( 25 ¢ ) Half dollar ( 50 ¢ ) Dollar ( $1 ) Bullion coinage America the Beautiful Silver Bullion Coins American Buffalo American Gold
Eagle American Platinum Eagle American Silver Eagle American Palladium Eagle Current paper money $1 $2 $5 $10 $20 $50 $100 See also Bicentennial coinage
Commemoratives Early Commemoratives Modern Commemoratives Confederate dollar Large denominations Obsolete denominations Promotional fake denominations
Counterfeit United States currency Mill Coin production In God We Trust E pluribus unum Annuit cœptis Nicknames Replacement banknote Sales tax token Retrieved
from `` https://en.wikipedia.org/w/index.
</item 6>

<item 7>
Director of the United States Mint
Hayes James P. Kimball July 1885 October 1889 Grover Cleveland Edward O. Leech October 1889 May 1893 Benjamin Harrison Robert E. Preston November 1893 February
1898 Grover Cleveland George E. Roberts February 1898 July 1907 William McKinley Frank A. Leach September 1907 August 1909 Theodore Roosevelt A. Piatt Andrew
November 1909 June 1910 William Howard Taft George E. Roberts July 1910 November 1914 Robert W. Woolley March 1915 July 1916 Woodrow Wilson F.H.
</item 7>

<item 8>
Director of the United States Mint
List of Directors of the United States Mint ( edit ) Name Assumed Office Left Office President Appointed By David Rittenhouse April 1792 June 1795 George
Washington Henry William de Saussure July 1795 October 1795 Elias Boudinot October 1795 July 1805 Robert Patterson January 1806 July 1824 Thomas Jefferson
Samuel Moore July 1824 July 1835 James Monroe Robert M. Patterson May 1835 July 1851 Andrew Jackson George Nicholas Eckert July 1851 April 1853 Millard Fillmore
Thomas M. Pettit April 1853 May 1853 Franklin Pierce James Ross Snowden June 1853 May 1861 James Pollock May 1861 September 1866 Abraham Lincoln William
Millward October 1866 April 1867 Andrew Johnson Henry Linderman April 1867 May 1869 James Pollock May 1869 March 1873 Ulysses S. Grant Henry Linderman April
1873 December 1878 Horatio C. Burchard February 1879 June 1885 Rutherford B. Hayes James P. Kimball July 1885 October 1889 Grover Cleveland Edward O.
</item 8>

<item 9>
Director of the United States Mint
US Mint . Retrieved 13 April 2017 . Retrieved from `` https://en.wikipedia.org/w/index.php?title=Director_of_the_United_States_Mint&oldid=836791366 ''
Categories : Directors of the United States Mint United States Department of the Treasury 1792 establishments in the United States Talk About Wikipedia Add
links This page was last edited on 16 April 2018 , at 21 : 26 . About Wikipedia
</item 9>

<item 10>
Director of the United States Mint
Woolley March 1915 July 1916 Woodrow Wilson F.H. von Engelken September 1916 March 1917 Raymond T. Baker March 1917 March 1922 Frank Edgar Scobey March 1922
September 1923 Warren G. Harding Robert J. Grant November 1923 May 1933 Calvin Coolidge Nellie Tayloe Ross May 1933 April 1953 Franklin D. Roosevelt William H.
Brett July 1954 January 1961 Dwight D. Eisenhower Eva Adams October 1961 August 1969 John F. Kennedy Mary Brooks September 1969 February 1977 Richard Nixon
Stella Hackel Sims November 1977 April 1981 Jimmy Carter Donna Pope July 1981 August 1991 Ronald Reagan David J. Ryder September 1992 November 1993 George H.W.
</item 10>
=== END ITEM id=0d7c74b46f1ed3d1 ===

=== ITEM id=8b2279a51398b4e0 ===
Gold sentences and facts:
Sentence 0: In 1908 , the first five - day workweek in the United States was instituted by a New England cotton mill so that Jewish workers would not have to work on the Sabbath from sundown Friday to sundown Saturday . In 1926 , Henry Ford began shutting down his automotive factories for all of Saturday and Sunday . In 1929 , the Amalgamated Clothing Workers of America Union was the first union to demand a five - day workweek and receive it . After that , the rest of the United States slowly followed , but it was not until 1940 , when a provision of the 1938 Fair Labor Standards Act mandating a maximum 40 - hour workweek went into effect , that the two - day weekend was adopted nationwide .
  Fact 0.0: In 1908, the first five-day workweek in the United States was instituted by a New England cotton mill.
  Fact 0.1: The New England cotton mill instituted the five-day workweek so that Jewish workers would not have to work on the Sabbath from sundown Friday to sundown Saturday.
  Fact 0.2: In 1926, Henry Ford began shutting down his automotive factories for all of Saturday and Sunday.
  Fact 0.3: In 1929, the Amalgamated Clothing Workers of America Union was the first union to demand a five-day workweek.
  Fact 0.4: In 1929, the Amalgamated Clothing Workers of America Union received a five-day workweek.
  Fact 0.5: After that, the rest of the United States slowly followed.
  Fact 0.6: The two-day weekend was not adopted nationwide until 1940.
  Fact 0.7: The two-day weekend was adopted nationwide in 1940, when a provision of the 1938 Fair Labor Standards Act mandating a maximum 40-hour workweek went into effect.

Retrieved items:
<item 1>
Workweek and weekend
Of the remaining 50 % , 23 % have taken steps to reform their 6 - day workweek through such measures as reducing the working week from 6 days to 5.5 days .
United states ( edit ) The standard working week in the United States begins on Monday and ends on Friday , 40 hours per week , with Saturday and Sunday being
weekend days . However , in practice , only 42 % of employees work 40 - hour weeks . The average workweek for full - time employees is 47 hours . Most stores
are open for business on Saturday and often on Sunday as well , except in a few places where prohibited by law ( see Blue law ) . Increasingly , employers are
offering compressed work schedules to employees . Some government and corporate employees now work a 9 / 80 work schedule ( 80 hours over 9 days during a two -
week period ) -- commonly 9 hour days Monday to Thursday , 8 hours on one Friday , and off the following Friday . Jobs in healthcare , law enforcement ,
transportation , retail , and other service positions commonly require employees to work on the weekend or to do shift work .
</item 1>

<item 2>
Workweek and weekend
In 1908 , the first five - day workweek in the United States was instituted by a New England cotton mill so that Jewish workers would not have to work on the
Sabbath from sundown Friday to sundown Saturday . In 1926 , Henry Ford began shutting down his automotive factories for all of Saturday and Sunday . In 1929 ,
the Amalgamated Clothing Workers of America Union was the first union to demand a five - day workweek and receive it . After that , the rest of the United
States slowly followed , but it was not until 1940 , when a provision of the 1938 Fair Labor Standards Act mandating a maximum 40 - hour workweek went into
effect , that the two - day weekend was adopted nationwide . Over the succeeding decades , particularly in the 1940s , 1950s , and 1960s , an increasing number
of countries adopted either a Friday -- Saturday or Saturday -- Sunday weekend to harmonize with international markets .
</item 2>

<item 3>
Workweek and weekend
( February 4 , 2010 ) `` Shorter Workweek in a Tough Economy '' Archived November 28 , 2016 , at the Wayback Machine . Wisconsin Employment Law Letter (
hrhero.com ) Jump up ^ Rones et al. ( 1997 ) `` Trends in Hours of Work since the Mid-1970s '' Archived June 24 , 2016 , at the Wayback Machine . Monthly Labor
Review 120 ( 3 ) : 3 -- 12 Jump up ^ Coote , Anna ; Franklin , Jane ; Simms , Andrew ( February 2010 ) . `` 21 hours : Why a shorter working week can help us
all to flourish in the 21st century '' ( PDF ) . London : New Economics Foundation . ISBN 978 - 1 - 904882 - 70 - 1 . Archived from the original ( PDF ) on 9
February 2016 . Retrieved 18 October 2016 . Jump up ^ Stuart , H.
</item 3>

<item 4>
Workweek and weekend
In Ancient Rome , every eight days there was a nundinae . It was a market day , during which children were exempted from school and plebs ceased from work in
the field and came to the city to sell the produce of their labor or practice religious rites. . The French Revolutionary Calendar had ten - day weeks ( called
décades ) and allowed décadi , one out of the ten days , as a leisure day . In cultures with a four - day week , the three Sabbaths derives from the culture 's
main religious tradition : Friday ( Muslim ) , Saturday ( Jewish ) , and Sunday ( Christian ) . The present - day concept of the relatively longer ' week - end
' first arose in the industrial north of Britain in the early part of nineteenth century and was originally a voluntary arrangement between factory owners and
workers allowing Saturday afternoon off from 2pm in agreement that staff would be available for work sober and refreshed on Monday morning . The Oxford English
Dictionary traces the first use of the term weekend to the British magazine Notes and Queries in 1879 .
</item 4>

<item 5>
Workweek and weekend
1 Soviet Union 3.17 Singapore 3.18 South Africa 3.19 Thailand 3.20 United States 3.21 Vietnam 4 See also 5 References History ( edit ) Part of a series on
Organised labour Labour movement ( show ) Timeline New Unionism Proletariat Social Movement Unionism Socialism Syndicalism Anarcho - syndicalism Labour rights (
hide ) Child labour Eight - hour day Five - day work week Annual leave Collective bargaining Minimum wage Occupational safety and health Trade unions ( show )
Trade unions by country Trade union federations International comparisons ITUC IWA WFTU Strike action ( show ) Chronological list of strikes General strike
Secondary action Sitdown strike Work - to - rule Labour parties ( show ) Australian British Dutch Irish Israeli New Zealand Norwegian List of Labour parties
Academic disciplines ( show ) Industrial relations Labour economics Labour history Labour law This section needs expansion . You can help by adding to it . (
September 2013 ) A continuous seven - day cycle that runs throughout history paying no attention whatsoever to the phases of the moon , having a fixed day of
rest , was probably first practiced in Judaism , dated to the 6th century BC at the latest .
</item 5>

<item 6>
Workweek and weekend
In 1926 , Henry Ford standardized on a five - day workweek , instead of the prevalent six days , without reducing employees ' pay . Hours worked stabilized at
about 49 per week during the 1920s , and during the Great Depression fell below 40 . During the Depression , President Herbert Hoover called for a reduction in
work hours in lieu of layoffs . Later , President Franklin Roosevelt signed the Fair Labor Standards Act of 1938 , which established a five - day , 40 - hour
workweek for many workers . The proportion of people working very long weeks has since risen , and the full - time employment of women has increased
dramatically . The New Economics Foundation has recommended moving to a 21 - hour standard workweek to address problems with unemployment , high carbon
emissions , low well - being , entrenched inequalities , overworking , family care , and the general lack of free time . The Center for Economic and Policy
Research states that reducing the length of the work week would slow climate change and have other environmental benefits . Around the world ( edit ) (
Countries listed alphabetically . Some countries have Saturday a normal school day .
</item 6>

<item 7>
Workweek and weekend
Retail businesses mostly operate Monday through Saturday , with larger establishments being open seven days a week . China ( edit ) In China , there is a five -
day Monday - Friday working week , prior to which work on Saturday was standard . China began the two - day Saturday -- Sunday weekend on May 1 , 1995 . Most
government employees work 5 days a week ( including officials and industrial management ) . Most manufacturing facilities operate on Saturdays as well . However
, most shops , museums , cinemas and commercial establishments open on Saturdays , Sundays and holidays . Banks are also open throughout the weekend and on most
public holidays . During the period of public holidays , swapped holidays are common between the actual holiday and weekend , so a three - day or seven - day
holiday periods are created . The nearby Saturday or Sunday may be changed to a normal working day . For example , on a three - day holiday period , if the
actual holiday falls on a Tuesday , Monday will be swapped as a holiday , and citizens are required to work on the previous Saturday .
</item 7>

<item 8>
Workweek and weekend
In some Christian traditions , Sunday is the `` day of rest and worship '' . Jewish Shabbat or Biblical Sabbath lasts from sunset on Friday to the fall of full
darkness on Saturday ; as a result , the weekend in Israel is observed on Friday -- Saturday . Some Muslim - majority countries historically had a Thursday --
Friday or Friday -- Saturday weekend ; however , recently many such countries have shifted from Thursday -- Friday to Friday -- Saturday , or to Saturday --
Sunday . The Christian Sabbath was just one day each week , but the preceding day ( the Jewish Sabbath ) came to be taken as a holiday as well in the twentieth
century . This shift has been accompanied by a reduction in the total number of hours worked per week , following changes in employer expectations . The present
- day concept of the ' week - end ' first arose in the industrial north of Britain in the early part of nineteenth century and was originally a voluntary
arrangement between factory owners and workers allowing Saturday afternoon off from 2pm in agreement that staff would be available for work sober and refreshed
on Monday morning . The Amalgamated Clothing Workers of America Union was the first to successfully demand a five - day work week in 1929 .
</item 8>

<item 9>
Workweek and weekend
no show Sabbatical Sick leave Time clock Schedules Four - day week Eight - hour day Flextime Overtime Retroactive overtime Shift work Telecommuting Working time
Workweek and weekend Wages and salaries Income bracket Income tax Living wage Maximum wage National average salary World Europe Minimum wage Canada Hong Kong
Europe United States Progressive wage Singapore Overtime rate Paid time off Performance - related pay Salary Salary cap Working poor Benefits Annual leave
Casual Friday Day care Disability insurance Health insurance Life insurance Parental leave Pension Sick leave Take - home vehicle Safety and health Epilepsy and
employment Human factors and ergonomics Industrial noise Karōshi Protective clothing Occupational burnout Occupational disease Occupational exposure limit
Occupational health psychology Occupational injury Occupational stress Repetitive strain injury Sick building syndrome Work accident Occupational fatality
Workers ' compensation Workplace phobia Workplace wellness Equality Affirmative action Equal pay for women Gender pay gap Glass ceiling Infractions Corporate
abuse Accounting scandals Corporate behaviour Corporate crime Control fraud Corporate scandals Discrimination Dress code Employee handbook Employee monitoring
Evaluation Labour law Sexual harassment Sleeping while on duty Wage theft Whistleblower Workplace bullying Workplace harassment Workplace incivility Willingness
Boreout Civil conscription Conscription Dead - end job Extreme careerism Job satisfaction Organizational commitment McJob Refusal of work Slavery
</item 9>

<item 10>
Workweek and weekend
A number of provinces and municipalities across China , including Hebei , Jiangxi and Chongqing , have issued new policies , calling on companies to create 2.5
- day weekends . Under the plan , government institutions , state - owned companies , joint - ventures and privately held companies are to be given incentives
to allow their workers to take off at noon on Friday before coming back to the office on Monday . Hong Kong ( edit ) In Hong Kong , a typical working week for
local enterprises begins on Monday and ends at 1pm on Saturday , although most employees have alternate Saturdays off . After the introduction of the five - day
working week for the majority of government departments in 2006 , most multinational enterprises and large local companies followed suit , extended the working
day from 9am to 6pm so as to adopt a five - day work week . Despite the aforementioned official hours , many employees work overtime , and in the case of the
financial industry in particular , working 12 - hour days on a chronic basis is not uncommon . Most commercial establishments in the retail sector such as
restaurants , shops and cinemas , as well as public venues such as museums and libraries are open on Saturdays , Sundays and most public holidays .
</item 10>
=== END ITEM id=8b2279a51398b4e0 ===
