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

This file has 7 items.

=== ITEM id=7598d690b736ffd0 ===
Gold sentences and facts:
Sentence 0: The poem survives in a single manuscript , the Cotton Nero A.x. , which also includes three religious narrative poems : Pearl , Purity and Patience . All are thought to have been written by the same unknown author , dubbed the `` Pearl Poet '' or `` Gawain Poet '' , since all four are written in a North West Midland dialect of Middle English .
  Fact 0.0: The poem survives in a single manuscript, the Cotton Nero A.x.
  Fact 0.1: The Cotton Nero A.x. manuscript also includes three religious narrative poems: Pearl, Purity and Patience.
  Fact 0.2: Pearl, Purity and Patience are thought to have been written by the same unknown author.
  Fact 0.3: The unknown author is dubbed the 'Pearl Poet' or 'Gawain Poet'.
  Fact 0.4: The 'Pearl Poet' or 'Gawain Poet' name is given because all four poems are written in a North West Midland dialect of Middle English.
  Fact 0.5: All four poems are written in a North West Midland dialect of Middle English.

Retrieved items:
<item 1>
Sir Gawain and the Green Knight
6 Homoerotic interpretations 8 Modern adaptations 8.1 Books 8.2 Film and television 8.3 Theatre 8.4 Opera 9 References 10 External links Synopsis ( edit ) Sir
Gawain and the Green Knight ( from original manuscript , artist unknown ) In Camelot on New Year 's Day , King Arthur 's court is exchanging gifts and waiting
for the feasting to start when the king asks first to see or hear of an exciting adventure . A gigantic figure , entirely green in appearance and riding a green
horse , rides unexpectedly into the hall . He wears no armour but bears an axe in one hand and a holly bough in the other . Refusing to fight anyone there on
the grounds that they are all too weak to take him on , he insists he has come for a friendly `` Christmas game '' : someone is to strike him once with his axe
on condition that the Green Knight may return the blow in a year and a day . The splendid axe will belong to whoever takes him on .
</item 1>

<item 2>
Sir Gawain and the Green Knight
Sir Gawain and the Green Knight - wikipedia Sir Gawain and the Green Knight Jump to : navigation , search For the 1973 film adaptation , see Gawain and the
Green Knight ( film ) . Sir Gawain and the Green Knight First page of survived manuscript , circa 14th century . Author Anonymous Country Kingdom of England
Language Middle English Genre Poem , chivalric romance , Arthurian and alliterative verse . Publication date 14th century Sir Gawain and the Green Knight (
Middle English : Sir Gawayn and þe Grene Knyȝt ) is a late 14th - century Middle English chivalric romance . It is one of the best known Arthurian stories ,
with its plot combining two types of folklore motifs , the beheading game and the exchange of winnings . The Green Knight is interpreted by some as a
representation of the Green Man of folklore and by others as an allusion to Christ . Written in stanzas of alliterative verse , each of which ends in a rhyming
bob and wheel , it draws on Welsh , Irish , and English stories , as well as the French chivalric tradition .
</item 2>

<item 3>
Sir Gawain and the Green Knight
722 . ^ Jump up to : Hulbert , J.R. `` Syr Gawayn and the Grene Knyzt ( Concluded ) '' . Modern Philology. ( April 1916 ) 13.12 pp. 689 -- 730 . Jump up ^
Jackson , I. `` Sir Gawain 's Coat of Arms . '' The Modern Language Review . ( January 1920 ) 15.1 pp. 77 -- 79 . Jump up ^ Arthur , pp. 33 -- 34 . ^ Jump up to
: Arthur , p. 34 . Jump up ^ Arthur , p. 35 . Jump up ^ Cooke , Jessica ( 1998 ) . `` The Lady 's ' Blushing ' Ring in Sir Gawain and the Green Knight '' . The
Review of English Studies . 49 ( 193 ) : 1 -- 8 . doi : 10.1093 / res / 49.193. 1 . Jump up ^ Andrew , Malcolm ; Ronald Waldron ( 1996 ) . The Poems of the
Pearl Manuscript .
</item 3>

<item 4>
Sir Gawain and the Green Knight
`` King Arthur and King Cornwall '' . In Sir Gawain : Eleven Romances and Tales . Western Michigan University Medieval Institute Publications , 2000 . p. 427 .
ISBN 1 - 879288 - 59 - 1 ^ Jump up to : Lasater , Alice E. Spain to England : A Comparative Study of Arabic , European , and English Literature of the Middle
Ages . University Press of Mississippi , 1974 . ISBN 0 - 87805 - 056 - 6 Jump up ^ Rix , Michael M. `` A Re-Examination of the Castleton Garlanding '' .
Folklore . ( June 1953 ) 64.2 pp. 342 -- 344 Jump up ^ Buchanan , Alice ( June 1932 ) . `` The Irish Framework of Sir Gawain and the Green Knight '' . PLMA. 47
( 2 ) : 315 -- 338 . doi : 10.2307 / 457878 . Jump up ^ . Friedman , Albert B. Morgan le Fay in Sir Gawain and the Green Knight . Medieval Academy of America .
</item 4>

<item 5>
Sir Gawain and the Green Knight
Retrieved 2016 - 03 - 01 . External links ( edit ) Find more aboutSir Gawain and the Green Knightat Wikipedia 's sister projects Definitions from Wiktionary
Media from Wikimedia Commons News from Wikinews Quotations from Wikiquote Texts from Wikisource Learning resources from Wikiversity Online texts High -
resolution , full - sized scan of entire manuscript Sir Gawain and the Green Knight at Archive.org JRR Tolkien and EV Gordon 's Edition The poem in Middle
English Sir Gawain and the Green Knight public domain audiobook at LibriVox ( various translations ) Google Books results General information The Gawain / Pearl
Poet from the University of Calgary The Camelot Project Info on Sir Gawain Luminarium SGGK Website Oxford Bibliographies : Bibliography on Sir Gawain and the
Green Knight BBC documentary tour of locations thought to be found in the poem ... Holywell , the River Mersey , the ' Wild Wirral ' , the Peak District , the
Roaches , and Lud Chapel Sir Gawain Introduction - Article introducing various translations and adaptations of Sir Gawain The Pearl Poet Works Sir Gawain and
the Green Knight Pearl Patience Cleanness St.
</item 5>

<item 6>
Sir Gawain and the Green Knight
( 1980 ) . Christopher Tolkien , ed . Sir Gawain and the Green Knight , Pearl , and Sir Orfeo . Random House . ISBN 978 - 0 - 345 - 27760 - 2 . Jump up ^ White
, Michael ( 2003 ) . Tolkien : A Biography . New American Library . ISBN 0 - 451 - 21242 - 8 . Jump up ^ Farley , Frank E. ( 1930 ) . `` Rev. of Banks , Sir
Gawain , and Andrew , Sir Gawain '' ( PDF ) . Speculum. 5 ( 2 ) : 222 -- 24 . doi : 10.2307 / 2847870 . JSTOR 2847870 . Jump up ^ Baragona , Alan ( 2012 ) . ``
Rev. of Howes , Borroff , Sir Gawain and the Green Knight '' . Journal of English and Germanic Philology. 111 ( 4 ) : 535 -- 38 . doi : 10.5406 / jenglgermphil.
</item 6>

<item 7>
Sir Gawain and the Green Knight
JSTOR 462158 . Jump up ^ Arthur , pp. 121 -- 123 . Jump up ^ Frye , Northrop . Anatomy of Criticism , p. 186 . ISBN 0 - 691 - 01298 - 9 ^ Jump up to : Markman ,
Alan M. `` The Meaning of Sir Gawain and the Green Knight '' . PMLA . ( September 1957 ) 72.4 pp. 574 -- 586 Jump up ^ Sir Gawain and the Green Knight , second
edition . Ed . J.R.R. Tolkien and E.V. Gordon , 1925 , note to lines 2514ff . ASIN B000IPU84U Jump up ^ The Norton Anthology of English Literature , eighth
edition , Vol . BEd Stephen Greenblatt . New York , London : W.W. Norton and Co. , 2006 , p. 213 ( footnote ) . ISBN 0 - 393 - 92833 - 0 Jump up ^ Sir Gawain
and the Green Knight , line 2529 .
</item 7>

<item 8>
Sir Gawain and the Green Knight
doi : 10.5406 / jenglgermphil. 111.4. 0535 . Jump up ^ Hirsch , Edward ( 16 December 2007 ) . `` A Stranger in Camelot '' . The New York Times . p. 7.1 .
Retrieved 16 March 2010 . Jump up ^ Armitage , Simon ( 2007 ) . Sir Gawain and the Green Knight : A New Verse Translation . New York : Norton . ISBN 978 - 0 -
393 - 06048 - 5 . Jump up ^ Mills , Maldwyn ( 1991 ) . `` Review '' . The Yearbook of English Studies . 21 : 336 -- 337 . Jump up ^ `` Sir Gawain and the Green
Knight '' . BBC Four . 17 August 2010 . Retrieved 5 August 2012 . Jump up ^ Stone , Brian ( 1974 ) . Sir Gawain and the Green Knight . Penguin . p. 138 .
</item 8>

<item 9>
Sir Gawain and the Green Knight
Online : The Carle of Carlisle . Jump up ^ Kittredge , George Lyman . A Study of Sir Gawain and the Green Knight . Harvard University Press , 1960 . p. 76 .
ASIN B0006AWBXS Jump up ^ Kittredge , p. 83 Jump up ^ `` Sir Gawain and the Green Knight '' . Sir Gawain and the Green Knight , Pearl , Sir Orfeo . Translated
by J.R.R. Tolkien , New York : Ballantine Books , 1975 . 92 . Jump up ^ Burrow , J.A. A Reading of Sir Gawain and the Green Knight . London : Kegan Paul Ltd. ,
1965 . p. 162 . ISBN 0 - 7100 - 8695 - 4 ^ Jump up to : Burnley , J.D. ( 1973 ) . `` The Hunting Scenes in ' Sir Gawain and the Green Knight ' `` '' . The
Yearbook of English Studies . 3 : 1 -- 9 .
</item 9>

<item 10>
Sir Gawain and the Green Knight
Erkenwald Characters Green Knight Lady Bertilak Poetic style Bob and wheel Alliterative verse Wikiquote Wikisource texts ( hide ) Sir Gawain and the Green
Knight by Pearl Poet Characters Green Knight Gawain Lady Bertilak Guinevere Morgan le Fay Caradoc Film Gawain and the Green Knight ( 1973 ) Sword of the Valiant
( 1984 ) Literature The Squire , His Knight , and His Lady The Hollowing Opera Gawain Related The Greene Knight Honi soit qui mal y pense VIAF : 174581151 LCCN
: n79068347 GND : 4125015 - 1 SUDOC : 028245075 BNF : cb12012068r ( data ) Retrieved from `` https://en.wikipedia.org/w/index.
</item 10>
=== END ITEM id=7598d690b736ffd0 ===

=== ITEM id=31bdb4bff9d65c77 ===
Gold sentences and facts:
Sentence 0: Downton Abbey is a British period drama television series created by Julian Fellowes and co-produced by Carnival Films and Masterpiece . It first aired on ITV in the United Kingdom on 26 September 2010 , and on PBS in the United States on 9 January 2011 , as part of the Masterpiece Classic anthology . Six series have been made , the sixth airing in the autumn of 2015 in the UK and Ireland , and in January 2016 in the United States . On 26 March 2015 , the sixth series was confirmed to be the final series , with the final episode airing in the UK on 25 December 2015 on ITV . During the course of the programme , 52 episodes of Downton Abbey aired over six series .
  Fact 0.0: Downton Abbey is a British period drama television series.
  Fact 0.1: Downton Abbey was created by Julian Fellowes.
  Fact 0.2: Downton Abbey was co-produced by Carnival Films and Masterpiece.
  Fact 0.3: Downton Abbey first aired on ITV in the United Kingdom on 26 September 2010.
  Fact 0.4: Downton Abbey first aired on PBS in the United States on 9 January 2011.
  Fact 0.5: The PBS airing was part of the Masterpiece Classic anthology.
  Fact 0.6: Six series of Downton Abbey have been made.
  Fact 0.7: The sixth series aired in the autumn of 2015 in the UK and Ireland.
  Fact 0.8: The sixth series aired in January 2016 in the United States.
  Fact 0.9: On 26 March 2015, the sixth series was confirmed to be the final series.
  Fact 0.10: The final episode of Downton Abbey aired in the UK on 25 December 2015 on ITV.
  Fact 0.11: 52 episodes of Downton Abbey aired over six series.

Retrieved items:
<item 1>
List of Downton Abbey episodes
List of Downton Abbey episodes - Wikipedia List of Downton Abbey episodes Jump to : navigation , search This article 's plot summaries may be too long or
excessively detailed . Please help improve them by removing unnecessary details and making them more concise . ( January 2016 ) ( Learn how and when to remove
this template message ) Downton Abbey is a British period drama television series created by Julian Fellowes and co-produced by Carnival Films and Masterpiece .
It first aired on ITV in the United Kingdom on 26 September 2010 , and on PBS in the United States on 9 January 2011 , as part of the Masterpiece Classic
anthology . Six series have been made , the sixth airing in the autumn of 2015 in the UK and Ireland , and in January 2016 in the United States . On 26 March
2015 , the sixth series was confirmed to be the final series , with the final episode airing in the UK on 25 December 2015 on ITV . During the course of the
programme , 52 episodes of Downton Abbey aired over six series .
</item 1>

<item 2>
List of Downton Abbey episodes
Michael Gregson is about to depart for Germany ; if he becomes a German citizen , he may be able to divorce his insane wife . Lady Rosamund discovers that Edith
spent the night with Gregson , and warns her about damaging her reputation . Lord Gillingham follows Mary back from London and proposes marriage . She refuses ,
saying she is still grieving Matthew 's death . 30 5 `` Episode Five '' Philip John Julian Fellowes 20 October 2013 ( 2013 - 10 - 20 ) 11.39 May 1922 . Alfred
applies to a chef training program at The Ritz , under Escoffier , but is turned down . Carson considers re-hiring Mr Molesley if Alfred leaves , but Molesley ,
unhappy at being demoted to a footman , dithers too long over the offer , and loses the opportunity . Thomas pressures Cora 's new lady 's maid , Baxter , to
spy on everyone for him . When a tenant farmer dies owing significant back rent , Mary and Tom want to end the lease and run the farm themselves .
</item 2>

<item 3>
List of Downton Abbey episodes
overall No. in series Title Directed by Written by Original air date UK viewers ( millions ) 26 `` Episode One '' David Evans Julian Fellowes 22 September 2013
( 2013 - 09 - 22 ) 11.96 February 1922 . Six months on , a distraught Lady Mary mourns Matthew , and feels little motherly love for baby George , now six months
old . Lord Grantham and Branson dispute the management of the estate and struggle to pay death duties . Miss O'Brien suddenly leaves to serve Lady Flintshire ,
on her way to India . Needing a maid in a hurry , Cora hires former maid Edna , who has trained as a lady 's maid , on the strength of the reference which Tom
persuaded Mrs Hughes to give her ; Carson and Mrs Hughes see trouble ahead , and warn Tom to keep his distance . Miss West , the nanny of George and Sybbie ,
Mary 's son and Tom 's daughter , respectively , acts insolently with Thomas , and the underbutler lets Cora know of her tendency to leave them to their own
devices .
</item 3>

<item 4>
List of Downton Abbey episodes
45 Christmas special No . overall Title Directed by Written by Original air date UK viewers ( millions ) 16 `` Christmas at Downton Abbey '' Brian Percival
Julian Fellowes 25 December 2011 ( 2011 - 12 - 25 ) 12.11 Series 3 ( 2012 ) ( edit ) ITV confirmed the commissioning of a third series on 3 November 2011 . It
entered production in early 2012 , and began airing 16 September 2012 on ITV and aired in the United States beginning 6 January 2013 . Series 3 No . overall No.
in series Title Directed by Written by Original air date UK viewers ( millions ) 17 `` Episode One '' Brian Percival Julian Fellowes 16 September 2012 ( 2012 -
09 - 16 ) 11.43 7.9 ( PBS ) March / April 1920 . Cora 's mother , Martha Levinson , sweeps into Downton for Matthew and Mary 's wedding , aggravating the
Dowager Countess ( among others ) with her critical and `` revolutionary '' American way of seeing things .
</item 4>

<item 5>
List of Downton Abbey episodes
Bates and Anna plan their future , and contemplate selling Bates 's mother 's house in London and buying a house or hotel nearby to let out . Daisy feels let
down by the Labour government , and is no longer keen to continue studying , but Molesley and Mr Mason persuade her otherwise . Tom has written to his cousin in
Boston about moving to the United States . Lord Merton 's two sons are rude and insolent to Isobel during dinner ; Larry , who had at an earlier dinner insulted
Tom and drugged his drink , sneers that Isobel is a middle - class woman who can not possibly succeed in her public role as his father 's wife . His angry
father orders him to leave . Atticus proposes to Rose . Robert 's beloved dog Isis is ill with cancer . He and Cora take her into their bed to comfort her in
her final hours .
</item 5>

<item 6>
List of Downton Abbey episodes
Christmas special No . overall Title Directed by Written by Original air date UK viewers ( millions ) 52 `` The Finale '' Michael Engler Julian Fellowes 25
December 2015 ( 2015 - 12 - 25 ) 11.22 29 September -- 31 December 1925 : Carson fumbles serving dinner when his hands begin shaking , suffering from an
unspecified hereditary disease his family called ' the palsy ' . Thomas secures a job with an elderly couple with a staff of only three , but is unhappy there .
Edith offers to expand Spratt 's magazine column to a whole page ; Denker discovers Spratt 's secret , and tries to get him dismissed by revealing it to the
Dowager Countess , but she finds the column jolly amusing and keeps him on . Molesley leaves service when the Headmaster offers him a teaching position , plus a
cottage ; upon parting , he and Baxter finally express their mutual romantic affection . Lord Merton tells Isobel he suffers from pernicious anaemia , but still
wants to marry her .
</item 6>

<item 7>
List of Downton Abbey episodes
Series 6 No . overall No. in series Title Directed by Written by Original air date UK viewers ( millions ) 44 `` Episode One '' Minkie Spiro Julian Fellowes 20
September 2015 ( 2015 - 09 - 20 ) 10.32 April 1925 . Big changes are afoot . Robert anticipates having to cut costs and reduce staff , worrying Carson . Mrs
Hughes , nervous she will be embarrassed when performing her wifely duties after marrying Carson , asks Mrs Patmore to sound him out ; after much embarrassment
, Carson says he expects them to have a full marriage , which reassures Mrs Hughes . The local hospital is threatened with a take - over by the larger County
Hospital ; the board is split , with Violet objecting , and Isobel approving in the name of efficiency . Rita Bevan , a former chambermaid at the hotel where
Mary and Lord Gillingham stayed , attempts to blackmail Mary by exposing their tryst .
</item 7>

<item 8>
List of Downton Abbey episodes
With the future of Downton hanging in the balance , Mary and the Dowager Countess conspire to show Downton at its grandest to persuade Mrs Levinson to give
Downton more money and save them from financial ruin . Their attempts are impeded by the oven breaking down , giving Mrs Levinson a chance to take over running
the party herself in radical fashion . She reveals , when finally asked for help , that though she will `` entertain them in New York '' she can not provide
them with money , as her own late husband tied it up in the estate , much as Violet 's did . She tells them they have to adapt to the changing world , and
states her intent to return to America . Meanwhile downstairs , O'Brien and Thomas wage war against one another due to Alfred 's recent promotion to valet to
Matthew , whilst Mrs Levinson 's maid takes a shine to Alfred . Mrs Hughes faces a cancer scare that she shares only with Mrs Patmore and Doctor Clarkson , and
Isobel Crawley finds that Ethel has become a prostitute .
</item 8>

<item 9>
List of Downton Abbey episodes
Molesley and Baxter take a photo of Bates to different pubs in York , and find the one where he had lunch on the day in question , proving he was not in London
when Green was killed . The witness who identified Anna becomes uncertain , and she is released on bail . A big Christmas party is held for the Downton
household . It also serves as a goodbye party for Tom , who has found work as a writer in America and will be taking his daughter Sybbie with him . Bates
returns and enjoys a tender reunion with Anna . During a private break , Violet confesses to Isobel that she and Prince Kuragan had attempted to run away
together . Series 6 ( 2015 ) ( edit ) The sixth series premiered on 20 September 2015 in the UK and on 3 January 2016 in the United States . This series was the
last , the 2015 Christmas special titled `` The Finale '' . Due to increasingly popular ratings worldwide , a film is said to be in production . Series 6 No .
overall No.
</item 9>

<item 10>
List of Downton Abbey episodes
33 11 `` Episode Four '' Brian Kelly Julian Fellowes 9 October 2011 ( 2011 - 10 - 09 ) 11.30 12 5 `` Episode Five '' Brian Kelly Julian Fellowes 16 October 2011
( 2011 - 10 - 16 ) 11.59 13 6 `` Episode Six '' Andy Goddard Julian Fellowes 23 October 2011 ( 2011 - 10 - 23 ) 11.33 14 7 `` Episode Seven '' James Strong
Julian Fellowes 30 October 2011 ( 2011 - 10 - 30 ) 12.26 15 8 `` Episode Eight '' James Strong Julian Fellowes 6 November 2011 ( 2011 - 11 - 06 ) 12.45
Christmas special No .
</item 10>
=== END ITEM id=31bdb4bff9d65c77 ===

=== ITEM id=7a2c2ca2ea9bac93 ===
Gold sentences and facts:
Sentence 0: `` Rip It Up '' was a 1983 single by Scottish post-punk band Orange Juice . It was the second single to be released from their 1982 album of the same name . The song became the band 's only UK top 40 success , reaching no . 8 in the chart . `` Rip It Up '' signalled a departure from the sound of the band 's earlier singles , with Chic - influenced guitars and using a synthesiser to create a more disco - oriented sound .
  Fact 0.0: 'Rip It Up' was a 1983 single by Scottish post-punk band Orange Juice.
  Fact 0.1: 'Rip It Up' was the second single released from Orange Juice's 1982 album of the same name.
  Fact 0.2: 'Rip It Up' became Orange Juice's only UK top 40 success.
  Fact 0.3: 'Rip It Up' reached no. 8 in the chart.
  Fact 0.4: 'Rip It Up' signalled a departure from the sound of the band's earlier singles.
  Fact 0.5: 'Rip It Up' had Chic-influenced guitars.
  Fact 0.6: 'Rip It Up' used a synthesiser to create a more disco-oriented sound.
Sentence 1: `` Rip It Up '' Single by Orange Juice from the album Rip It Up B - side `` Snake Charmer '' `` A Sad Lament '' Released February 1983 Format 7 '' 12 '' Recorded Berwick Street Studios , London Genre Post-punk Label Polydor Songwriter ( s ) Edwyn Collins Producer ( s ) Martin Hayles Orange Juice singles chronology `` I Ca n't Help Myself '' ( 1982 ) `` Rip It Up '' ( 1983 ) `` Flesh of My Flesh '' ( 1983 ) `` I Ca n't Help Myself '' ( 1982 ) `` Rip It Up '' ( 1983 ) `` Flesh of My Flesh '' ( 1983 )
  Fact 1.0: 'Rip It Up' is a single by Orange Juice from the album Rip It Up.
  Fact 1.1: The B-side of 'Rip It Up' is 'Snake Charmer'.
  Fact 1.2: The B-side of 'Rip It Up' is 'A Sad Lament'.
  Fact 1.3: 'Rip It Up' was released in February 1983.
  Fact 1.4: 'Rip It Up' was released in 7-inch and 12-inch formats.
  Fact 1.5: 'Rip It Up' was recorded at Berwick Street Studios, London.
  Fact 1.6: 'Rip It Up' is in the post-punk genre.
  Fact 1.7: 'Rip It Up' was released on the Polydor label.
  Fact 1.8: 'Rip It Up' was songwritten by Edwyn Collins.
  Fact 1.9: 'Rip It Up' was produced by Martin Hayles.
  Fact 1.10: Orange Juice's singles chronology lists 'I Ca n't Help Myself' (1982), 'Rip It Up' (1983) and 'Flesh of My Flesh' (1983).

Retrieved items:
<item 1>
Rip It Up (Orange Juice song)
Track listing ( edit ) Single 7 '' No . Title Writer ( s ) Length 1 . `` Rip It Up '' Edwyn Collins 3 : 51 2 . `` Snake Charmer '' Malcolm Ross 4 : 43 Double 7
'' No . Title Writer ( s ) Length 1 . `` Rip It Up '' Edwyn Collins 2 . `` Snake Charmer '' Malcolm Ross 3 . `` Love Sick '' ( Live ) Edwyn Collins 4 . `` A Sad
Lament '' Edwyn Collins 12 '' No . Title Writer ( s ) Length 1 . `` Rip It Up '' ( Long Version ) Edwyn Collins 2 .
</item 1>

<item 2>
Rip It Up (Orange Juice song)
Rip It Up ( Orange Juice song ) - wikipedia Rip It Up ( Orange Juice song ) Jump to : navigation , search `` Rip It Up '' Single by Orange Juice from the album
Rip It Up B - side `` Snake Charmer '' `` A Sad Lament '' Released February 1983 Format 7 '' 12 '' Recorded Berwick Street Studios , London Genre Post-punk
Label Polydor Songwriter ( s ) Edwyn Collins Producer ( s ) Martin Hayles Orange Juice singles chronology `` I Ca n't Help Myself '' ( 1982 ) `` Rip It Up '' (
1983 ) `` Flesh of My Flesh '' ( 1983 ) `` I Ca n't Help Myself '' ( 1982 ) `` Rip It Up '' ( 1983 ) `` Flesh of My Flesh '' ( 1983 ) `` Rip It Up '' was a 1983
single by Scottish post-punk band Orange Juice . It was the second single to be released from their 1982 album of the same name . The song became the band 's
only UK top 40 success , reaching no . 8 in the chart .
</item 2>

<item 3>
The Rolling Stones
He also encouraged the press to use provocative headlines such as `` Would you let your daughter marry a Rolling Stone ? '' , although Wyman says `` Our
reputation and image as the Bad Boys came later , completely there , accidentally ... ( Oldham ) never did engineer it . He simply exploited it exhaustively ''
. In a 1972 interview , Wyman stated `` We were the first pop group to break away from the whole Cliff Richard thing where the bands did little dance steps ,
wore identical uniforms and had snappy patter . '' A cover version of Chuck Berry 's `` Come On '' was the Rolling Stones ' first single , released on 7 June
1963 . The band refused to play it at live gigs , and Decca bought only one ad to promote the record . With Oldham 's direction , fan - club members bought
copies at record shops polled by the charts , helping `` Come On '' rise to No. 21 on the UK Singles Chart . Having a charting single gave the band entree to
play outside London , starting with a booking at the Outlook Club in Middlesbrough on 13 July , sharing the billing with the Hollies .
</item 3>

<item 4>
Rip It Up (Orange Juice song)
Discogs . Retrieved 26 May 2013 . ^ Jump up to : `` Orange Juice - Rip It Up ( Song ) '' . charts.org.nz . Retrieved 26 May 2013 . Jump up ^ `` Orange Juice -
Rip It Up '' . Discogs . Retrieved 26 May 2013 . Jump up ^ `` Orange Juice '' . Official Charts Company . Retrieved 26 May 2013 .
</item 4>

<item 5>
Rip It Up (Orange Juice song)
It marked a departure from their previous guitar - pop based material , instead utilising Chic style guitar - funk and a bubbling Roland TB - 303 synthesiser
bassline , becoming the first chart single to feature the instrument . The song also features a snatch of the guitar riff from `` Boredom '' , a song by
Buzzcocks that featured on their debut Spiral Scratch EP . The riff chimes briefly in , just as Collins namechecks the song in the lyrics claiming that `` ...
and my favourite song is entitled ' Boredom ' . '' Backing vocals on the song were provided by Paul Quinn , the lead singer of fellow Scottish band Bourgie
Bourgie , with whom Collins would later record a single in 1984 , a cover of the Velvet Underground song `` Pale Blue Eyes . '' Music video ( edit ) The video
opens with the band in a futuristic , but cheaply constructed , control room as they sing , dance and operate various controls . The band then watch themselves
on a monitor screen as they walk down a rainy British high street dressed in incongruous , brightly coloured summer clothes .
</item 5>

<item 6>
Rip It Up (Orange Juice song)
Official Charts Company . Retrieved 26 May 2013 . External links ( edit ) Music video on YouTube ( hide ) Orange Juice Edwyn Collins James Kirk David McClymont
Steven Daly Malcolm Ross Zeke Manyika Studio albums You Ca n't Hide Your Love Forever Rip It Up Texas Fever The Orange Juice Singles `` Falling and Laughing ''
`` Rip It Up '' Compilations Ostrich Churchyard The Heather 's on Fire The Glasgow School Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Rip_It_Up_(Orange_Juice_song)&oldid=751734896 '' Categories : 1983 singles Orange Juice ( band ) songs 1983 songs
Polydor Records singles Songs written by Edwyn Collins Hidden categories : Use dmy dates from May 2015 Use British English from May 2015 All articles with
unsourced statements Articles with unsourced statements from August 2013 Track listings with deprecated parameters Talk Contents About Wikipedia Add links This
page was last edited on 27 November 2016 , at 15 : 08 . About Wikipedia
</item 6>

<item 7>
Rip It Up (Orange Juice song)
8 in the chart . `` Rip It Up '' signalled a departure from the sound of the band 's earlier singles , with Chic - influenced guitars and using a synthesiser to
create a more disco - oriented sound . The song was sampled in 2009 by British soul singer Beverley Knight on her song `` In Your Shoes '' from the album 100 %
. In 2014 , NME ranked it at number 216 in its list of The 500 Greatest Songs of All Time . It was also included by Pitchfork at number 157 in a list of The
Best 200 Songs of the 1980s . Contents ( hide ) 1 Recording and influences 2 Music video 3 UK single release 3.1 Track listing 4 Chart positions 5 References 6
External links Recording and influences ( edit ) Roland TB - 303 The song was recorded as part of the sessions for Orange Juice 's second studio album and would
go on to become the title track of said album .
</item 7>

<item 8>
The Rolling Stones
Numbers from various concerts and rehearsals ( mostly acoustic ) made up Stripped ( UK 9 ; US 9 ) , which featured a cover of Bob Dylan 's `` Like a Rolling
Stone '' , as well as infrequently played songs like `` Shine a Light '' , `` Sweet Virginia '' and `` The Spider and the Fly '' . On 8 September 1994 , the
Stones performed their new song `` Love Is Strong '' as well as `` Start Me Up '' at the 1994 MTV Video Music Awards at Radio City Music Hall in New York . The
band received the Lifetime Achievement Award at the ceremony . The Rolling Stones were the first major recording artists to broadcast a concert over the
Internet ; a 20 - minute video was broadcast on 18 November 1994 using the Mbone at 10 frames per second . The broadcast , engineered by Thinking Pictures and
financed by Sun Microsystems , was one of the first demonstrations of streaming video ; while it was not a true webcast , it introduced many to the technology .
The Rolling Stones ended the 1990s with the album Bridges to Babylon ( UK 6 ; US 3 ) , released in 1997 to mixed reviews .
</item 8>

<item 9>
Rip It Up (Orange Juice song)
The video then cuts back to the control room , this time with the band playing their instruments superimposed over it , before returning to more scenes of a
British city in torrential rain as the band walk around in scuba diving gear . The video finally cuts back to the band playing in a silver foil covered room ,
before superimposing them over a pile of random photographs . UK single release ( edit ) `` Rip It Up '' was released as a single in the UK in February 1983 .
The seven inch vinyl version of the single was available in three versions , a double pack including a second seven - inch and a fold out poster , along with
two versions of the standard release , initially with a silver injection moulded labels , and then subsequently with paper printed labels . The song was also
released on twelve inch vinyl , with extended versions of the title track and B - side . All versions were housed in a paper sleeve depicting a US P - 40
Warhawk fighter plane ( decorated with eyes and teeth ) partially submerged , tail first , in the sea , drawn by Edwyn Collins . Track listing ( edit ) Single 7
'' No .
</item 9>

<item 10>
Rip It Up (Orange Juice song)
`` Rip It Up '' ( Long Version ) Edwyn Collins 2 . `` A Sad Lament '' ( Long Version ) Edwyn Collins Chart positions ( edit ) Chart ( 1983 ) Peak position UK
Singles Chart 8 New Zealand Singles Chart 42 References ( edit ) Jump up ^ http://www.rocklistmusic.co.uk/nme_500_greatest_songs_2014.htm Jump up ^
http://pitchfork.com/features/lists-and-guides/9700-the-200-best-songs-of-the-1980s/?page=3 ^ Jump up to : `` Buzzcocks : Boredom / Orange Juice : Rip It Up ''
. Stylus Magazine . 25 August 2004 . Retrieved 26 May 2013 . ^ Jump up to : `` Orange Juice - Rip It Up 12 '' . Discogs . Retrieved 26 May 2013 . Jump up ^ ``
Orange Juice - Rip It Up '' . Discogs . Retrieved 26 May 2013 . Jump up ^ `` Orange Juice - Rip It Up '' . Discogs . Retrieved 26 May 2013 .
</item 10>
=== END ITEM id=7a2c2ca2ea9bac93 ===

=== ITEM id=2add4c25e0447df5 ===
Gold sentences and facts:
Sentence 0: State Pre-Prohibition ( prior to 1919 ) Post-Prohibition ( after 1933 ) 1970s / 26th Amendment ( adopted in 1971 ) 1980s / Drinking Age Act of 1984 21st century Alabama Pre 1881 : None 1881 : 21 ( none with consent of parent ) 21 1975 : Lowered to 19 1986 : Raised to 21 21 ( no one underage is allowed consumption Section 28 - 1 ) Alaska ? 21 1970 : Lowered to 19 1984 : Raised to 21 with grandfather clause ( if born before January 1 , 1965 ) 21 Arizona ? 21 1972 : Lowered to 19 1985 : Raised to 21 21 Arkansas Pre 1925 : None 1925 : 21 21 21 21 21 California Pre 1891 : Regulated by municipality / county ( common age was 16 ) 1891 : 18 ( statewide ) 1933 : 21 21 21 21 ( except small amounts for religious ceremonies ) Colorado None 1945 : 18 : for beer 21 : for wine and liquor 18 Raised to 21 in 1987 21 Connecticut ? 21 1972 : Lowered to 18 1982 : Raised to 19 1983 : Raised to 20 1985 : Raised to 21 21 ( exemptions : ( 1 ) a person over age eighteen who is an employee or permit holder under section 30 - 90a and who possesses alcoholic liquor in the course of such person 's employment or business , ( 2 ) a minor who possesses alcoholic liquor on the order of a practicing physician , or ( 3 ) a minor who possesses alcoholic liquor while accompanied by a parent , guardian or spouse of the minor , who has attained the age of twenty - one . Nothing in this subsection shall be construed to burden a person 's exercise of religion under section 3 of article first of the Constitution of the state in violation of subsection ( a ) of section 52 - 571b . Delaware ? 21 1972 : Lowered to 20 1984 : Raised to 21 21 District of Columbia ? 18 : for beer and wine 21 : for liquor 18 : for beer and wine 21 : for liquor 1986 : Raised to 21 with grandfather clause . 21 Florida ? 18 1980 : 19 1985 : Raised to 21 with grandfather clause 21 Georgia ? 21 1972 : Lowered to 18 1984 : Raised to 19 1985 : Raised to 20 1986 : Raised to 21 21 Hawaii ? 21 1972 : Lowered to 18 1986 : Raised to 21 21 Idaho ? 20 : for beer 21 : for wine and liquor 1972 : Lowered to 19 ( all ) 1987 : Raised to 21 in 1987 ( Apr 11 ) with grandfather clause 21 Illinois 1872 : Age of majority ( none with written consent of parent ) pre 1961 : Age of majority 1961 : Raised to 21 1973 : Lowered to 19 : for beer and wine 21 : for liquor 1980 : Raised to 21 21 Indiana ? Post-1934 : 21 21 21 21 Iowa ? 21 1972 : Lowered to 19 in 1972 ( Jul 1 ) 1973 : Lowered to 18 in 1973 ( Jul 1 ) 1978 : Raised to 19 in 1978 ( Jul 1 ) 1986 : Raised to 21 in 1986 ( Jul 1 ) with grandfather clause . 21 Kansas ? 1949 : 18 : for beer and wine with less than 4 % ABV 21 : for liquor 18 : for beer and wine with less than 4 % ABV 21 : for beer and wine with 4 % ABV or more 21 : for liquor 1985 : Raised to 21 21 ( However , anyone under 21 may consume cereal malt beverages with parental supervision on their own property ) Kentucky ? 1938 : 21 21 21 21 Louisiana ? As of 1948 : 18 18 1987 : Raised to 21 de jure in 1987 , but de facto age was still 18 until 1995 due to a sale loophole . 1995 : De facto age raised to 21 in 1995 when loophole was closed . 1996 : In 1996 , briefly lowered by Louisiana Supreme Court to 18 until it reversed its decision , raising to 21 three months later . Other exceptions still remain . 21 ( Exceptions to state law include : For an established religious purpose ; When a person under twenty - one years of age is accompanied by a parent , spouse , or legal guardian twenty - one years of age or older ; For medical purposes when purchased as an over the counter medication , or when prescribed or administered by a licensed physician , pharmacist , dentist , nurse , hospital , or medical institution ; In a private residence , which shall include a residential dwelling and up to twenty contiguous acres , on which the dwelling is located , owned by the same person who owns the dwelling ; The sale , handling , transport , or service in dispensing of any alcoholic beverage pursuant to lawful ownership of an establishment or to lawful employment of a person under twenty - one years of age by a duly licensed manufacturer , wholesaler , or retailer of beverage alcohol . ) Maine ? 21 1969 : Lowered to 20 1972 : Lowered to 18 1977 : Raised to 20 1985 : Raised to 21 21 Maryland ? 21 1974 : Lowered to 18 : for beer and wine 21 : for liquor 1982 : Raised to 21 with grandfather clause 21 ( however anyone can drink when parents , spouses , teachers are present ) Massachusetts ? 21 1973 : Lowered to 18 1979 : Raised to 20 in 1979 ( Apr 16 ) Raised to 21 in 1985 21 Michigan ? 21 1972 : Lowered to 18 in 1972 ( Jan 1 ) 1978 : Raised to 19 in 1978 ( Dec 3 ) 1978 : Raised to 21 in 1978 ( Dec 21 ) , 18 days later . First state to raise age to 21 since it was lowered 21 Minnesota ? 21 1973 : Lowered to 18 ( June 1 ) 1976 : Raised to 19 ( Sept 1 ) First state to raise age after lowering . 1986 : Raised to 21 21 Mississippi ? 1966 : 18 : for beer and wine 21 : for liquor ( alcohol not legalized until 1966 ) 18 Raised to 21 in 1986 21 Missouri ? 1945 : 21 21 21 21 Montana ? 21 1971 : Lowered to 19 1973 : Lowered to 18 1979 : Raised to 19 1987 : Raised to 21 21 Nebraska ? 21 1969 : Lowered to 20 1972 : Lowered to 19 1980 : Raised to 20 1985 : Raised to 21 21 Nevada ? 1933 : 21 21 21 21 New Hampshire ? 21 1973 : Lowered to 18 1979 : Raised to 20 1985 : Raised to 21 21 New Jersey Before 1880 : None Post-1880 : 18 ( penalties only against businesses ) 21 1973 : Lowered to 18 1980 : Raised to 19 1983 : Raised to 21 21 New Mexico ? 1934 : 21 21 21 21 New York 21 18 18 1982 : Raised to 19 in 1982 ( Dec ) 1985 : Raised to 21 in 1985 ( Dec ) 21 North Carolina ? 18 18 : for beer and wine 21 : for liquor 1983 : 19 : for beer and wine 21 : for liquor 1986 : Raised to 21 in 1986 ( Sep 1 ) 21 North Dakota ? 1936 : 21 21 21 21 Ohio ? 1935 : 18 : for 3.2 % ABW beer 21 : for beer stronger than 3.2 % ABW , wine and liquor 18 : for 3.2 % ABW beer 21 : for beer stronger than 3.2 % ABW , wine and liquor 1982 : 19 : for 3.2 % ABW beer 21 : for beer stronger than 3.2 % ABW , wine and liquor 1982 : 19 : beer 21 : wine and liquor 1988 : 21 21 ( However , anyone can drink with parents or legal - age spouse ) Oklahoma ? Men : 21 Women : 18 : 3.2 % ABW beer 21 : for beer stronger than 3.2 % ABW , wine and liquor 1976 : Lowered to 18 ( for 3.2 % ABW beer ) for both sexes in 1976 ( Dec ) Last state to lower the drinking age . 1983 : Raised to 21 21 Oregon ? 1933 : 21 21 21 21 Pennsylvania ? 1935 : 21 21 21 21 Rhode Island ? 21 1970 : Lowered to 18 1980 : Raised to 19 1981 : Raised to 20 1984 : Raised to 21 21 South Carolina 21 1935 : 18 : for beer and wine 21 : for liquor 18 : for beer and wine 21 : for liquor 1984 : 19 : for beer and wine 21 : for liquor 1986 : Raised to 21 21 South Dakota ? 1934 : 21 ( all ) 1939 : Lowered to 18 : for 3.2 % beer Raised to 19 : for 3.2 % beer 1972 : 18 : for beer and wine 21 : for liquor 1984 : 19 : for beer and wine 21 : for liquor 1988 : Raised to 21 21 Tennessee Pre 1881 : None 1881 : 21 21 1964 : Lowered to 18 1979 : Raised to 19 1984 : Raised to 21 21 Texas Pre 1909 : 21 None ( with written permission of parent or guardian ) Post 1909 : 21 21 1973 : Lowered to 18 1981 : Raised to 19 1986 : Raised to 21 in 1986 ( Sep 1 ) 21 Utah ? 1935 : 21 21 21 21 Vermont ? 21 1971 : Lowered to 18 1986 : Raised to 21 21 Virginia ? 21 1974 : Lowered to 18 : for beer and wine ( 21 : for liquor remained ) 1981 : Raised to 19 for off - premises consumption 1983 : Raised to 19 ( all beer ) 1985 : Raised to 21 21 Washington `` Age of majority '' since 1877 21 21 21 21 West Virginia ? 18 : for beer and wine 21 : for liquor 1972 : Lowered to 18 ( all ) 1983 : Raised to 19 ( & 21 for non-residents ) 1986 : Raised to 21 21 Wisconsin 1839 -- 1866 : 18 : for wine and liquor None : for beer Post-1866 : 21 ( all ) 18 : for beer 21 : for wine and spirits 1957 : 21 : for residents of bordering states with age limit 21 . 1963 : Raised to 21 ( off - premises beer sales , remained 18 for on - premises ) . 1972 : Lowered to 18 ( all ) in 1972 ( Mar ) 1977 : Border state restriction lifted . 1984 : Raised to 19 1986 : Raised to 21 ( September 1 , 1986 ) 21 ( however anyone can drink when parents , spouses , teachers are present ) Wyoming ? 21 1973 : Lowered to 19 1988 : Raised to 21 in 1988 ( Jul 1 ) Last state to raise de jure drinking age to 21 21 American Samoa ? ? ? 21 ? 21 Northern Mariana Islands -- -- ? 21 ? 21 Puerto Rico ? 18 18 Kept at 18 despite 10 % highway funding penalty under Drinking Age Act . 18 Virgin Islands ? ? ? 18 18 Guam ? 21 ? 18 18 Raised to 21 in 2010
  Fact 0.0: The table lists each state's drinking age in five periods: pre-Prohibition (prior to 1919), post-Prohibition (after 1933), the 1970s (26th Amendment, adopted in 1971), the 1980s (Drinking Age Act of 1984), and the 21st century.
  Fact 0.1: Alabama had no drinking age before 1881.
  Fact 0.2: Alabama set its drinking age at 21 in 1881, with none with consent of parent.
  Fact 0.3: Alabama's drinking age was 21 after Prohibition.
  Fact 0.4: Alabama lowered its drinking age to 19 in 1975.
  Fact 0.5: Alabama raised its drinking age to 21 in 1986.
  Fact 0.6: Alabama's 21st-century drinking age is 21, and no one underage is allowed consumption under Section 28-1.
  Fact 0.7: Alaska's post-Prohibition drinking age was 21.
  Fact 0.8: Alaska lowered its drinking age to 19 in 1970.
  Fact 0.9: Alaska raised its drinking age to 21 in 1984, with a grandfather clause for those born before January 1, 1965.
  Fact 0.10: Alaska's 21st-century drinking age is 21.
  Fact 0.11: Arizona's post-Prohibition drinking age was 21.
  Fact 0.12: Arizona lowered its drinking age to 19 in 1972.
  Fact 0.13: Arizona raised its drinking age to 21 in 1985.
  Fact 0.14: Arizona's 21st-century drinking age is 21.
  Fact 0.15: Arkansas had no drinking age before 1925.
  Fact 0.16: Arkansas set its drinking age at 21 in 1925.
  Fact 0.17: Arkansas's drinking age stayed at 21 in the post-Prohibition, 1970s, 1980s and 21st-century periods.
  Fact 0.18: California was regulated by municipality or county before 1891, when the common age was 16.
  Fact 0.19: California set a statewide drinking age of 18 in 1891.
  Fact 0.20: California set its drinking age at 21 in 1933.
  Fact 0.21: California's drinking age stayed at 21 after 1933, except small amounts for religious ceremonies.
  Fact 0.22: Colorado had no drinking age before Prohibition.
  Fact 0.23: Colorado set its drinking age in 1945 at 18 for beer and 21 for wine and liquor.
  Fact 0.24: Colorado's drinking age was 18 in the 1970s.
  Fact 0.25: Colorado raised its drinking age to 21 in 1987.
  Fact 0.26: Colorado's 21st-century drinking age is 21.
  Fact 0.27: Connecticut's post-Prohibition drinking age was 21.
  Fact 0.28: Connecticut lowered its drinking age to 18 in 1972.
  Fact 0.29: Connecticut raised its drinking age to 19 in 1982.
  Fact 0.30: Connecticut raised its drinking age to 20 in 1983.
  Fact 0.31: Connecticut raised its drinking age to 21 in 1985.
  Fact 0.32: Connecticut's 21st-century drinking age is 21.
  Fact 0.33: Connecticut exempts a person over 18 who is an employee or permit holder under section 30-90a and who possesses alcoholic liquor in the course of such employment or business.
  Fact 0.34: Connecticut exempts a minor who possesses alcoholic liquor on the order of a practicing physician.
  Fact 0.35: Connecticut exempts a minor who possesses alcoholic liquor while accompanied by a parent, guardian or spouse of the minor who has attained the age of 21.
  Fact 0.36: Connecticut's age limit does not burden a person's exercise of religion under section 3 of article first of the state Constitution in violation of section 52-571b(a).
  Fact 0.37: Delaware's post-Prohibition drinking age was 21.
  Fact 0.38: Delaware lowered its drinking age to 20 in 1972.
  Fact 0.39: Delaware raised its drinking age to 21 in 1984.
  Fact 0.40: Delaware's 21st-century drinking age is 21.
  Fact 0.41: The District of Columbia set its drinking age at 18 for beer and wine and 21 for liquor.
  Fact 0.42: The District of Columbia's 1970s drinking age was 18 for beer and wine and 21 for liquor.
  Fact 0.43: The District of Columbia raised its drinking age to 21 in 1986, with a grandfather clause.
  Fact 0.44: The District of Columbia's 21st-century drinking age is 21.
  Fact 0.45: Florida's drinking age was 18 in the 1970s.
  Fact 0.46: Florida raised its drinking age to 19 in 1980.
  Fact 0.47: Florida raised its drinking age to 21 in 1985, with a grandfather clause.
  Fact 0.48: Florida's 21st-century drinking age is 21.
  Fact 0.49: Georgia's post-Prohibition drinking age was 21.
  Fact 0.50: Georgia lowered its drinking age to 18 in 1972.
  Fact 0.51: Georgia raised its drinking age to 19 in 1984.
  Fact 0.52: Georgia raised its drinking age to 20 in 1985.
  Fact 0.53: Georgia raised its drinking age to 21 in 1986.
  Fact 0.54: Georgia's 21st-century drinking age is 21.
  Fact 0.55: Hawaii's post-Prohibition drinking age was 21.
  Fact 0.56: Hawaii lowered its drinking age to 18 in 1972.
  Fact 0.57: Hawaii raised its drinking age to 21 in 1986.
  Fact 0.58: Hawaii's 21st-century drinking age is 21.
  Fact 0.59: Idaho's post-Prohibition drinking age was 20 for beer and 21 for wine and liquor.
  Fact 0.60: Idaho lowered its drinking age to 19 for all alcohol in 1972.
  Fact 0.61: Idaho raised its drinking age to 21 on April 11, 1987, with a grandfather clause.
  Fact 0.62: Idaho's 21st-century drinking age is 21.
  Fact 0.63: Illinois had a drinking age equal to the age of majority in 1872, with none with written consent of parent.
  Fact 0.64: Illinois's drinking age was the age of majority before 1961.
  Fact 0.65: Illinois raised its drinking age to 21 in 1961.
  Fact 0.66: Illinois lowered its drinking age to 19 for beer and wine in 1973, and kept 21 for liquor.
  Fact 0.67: Illinois raised its drinking age to 21 in 1980.
  Fact 0.68: Illinois's 21st-century drinking age is 21.
  Fact 0.69: Indiana's drinking age has been 21 since 1934.
  Fact 0.70: Indiana's drinking age was 21 in the 1970s, 1980s and 21st century.
  Fact 0.71: Iowa's post-Prohibition drinking age was 21.
  Fact 0.72: Iowa lowered its drinking age to 19 on July 1, 1972.
  Fact 0.73: Iowa lowered its drinking age to 18 on July 1, 1973.
  Fact 0.74: Iowa raised its drinking age to 19 on July 1, 1978.
  Fact 0.75: Iowa raised its drinking age to 21 on July 1, 1986, with a grandfather clause.
  Fact 0.76: Iowa's 21st-century drinking age is 21.
  Fact 0.77: Kansas set its post-Prohibition drinking age in 1949 at 18 for beer and wine under 4% ABV and 21 for liquor.
  Fact 0.78: Kansas's 1970s drinking age was 18 for beer and wine under 4% ABV, and 21 for beer and wine at 4% ABV or more and for liquor.
  Fact 0.79: Kansas raised its drinking age to 21 in 1985.
  Fact 0.80: Kansas's 21st-century drinking age is 21, but anyone under 21 may consume cereal malt beverages with parental supervision on their own property.
  Fact 0.81: Kentucky's drinking age was 21 from 1938 onward.
  Fact 0.82: Louisiana's drinking age was 18 as of 1948.
  Fact 0.83: Louisiana's drinking age was 18 in the 1970s.
  Fact 0.84: Louisiana's drinking age was raised to 21 de jure in 1987, but the de facto age stayed at 18 until 1995 because of a sale loophole.
  Fact 0.85: Louisiana's de facto drinking age was raised to 21 in 1995 when the loophole was closed.
  Fact 0.86: In 1996 the Louisiana Supreme Court briefly lowered the drinking age to 18, until it reversed its decision and raised it to 21 three months later.
  Fact 0.87: Louisiana's 21st-century drinking age is 21, with exceptions that remain.
  Fact 0.88: Louisiana's drinking age exception covers an established religious purpose.
  Fact 0.89: Louisiana's drinking age exception covers a person under 21 accompanied by a parent, spouse, or legal guardian who is 21 or older.
  Fact 0.90: Louisiana's drinking age exception covers medical purposes, when purchased as an over-the-counter medication or when prescribed or administered by a licensed physician, pharmacist, dentist, nurse, hospital, or medical institution.
  Fact 0.91: Louisiana's drinking age exception covers a private residence, including a residential dwelling and up to twenty contiguous acres on which the dwelling is located, owned by the same person who owns the dwelling.
  Fact 0.92: Louisiana's drinking age exception covers the sale, handling, transport, or service in dispensing of alcoholic beverages under lawful ownership of an establishment or lawful employment of a person under 21 by a duly licensed manufacturer, wholesaler, or retailer of beverage alcohol.
  Fact 0.93: Maine's post-Prohibition drinking age was 21.
  Fact 0.94: Maine lowered its drinking age to 20 in 1969.
  Fact 0.95: Maine lowered its drinking age to 18 in 1972.
  Fact 0.96: Maine raised its drinking age to 20 in 1977.
  Fact 0.97: Maine raised its drinking age to 21 in 1985.
  Fact 0.98: Maine's 21st-century drinking age is 21.
  Fact 0.99: Maryland's post-Prohibition drinking age was 21.
  Fact 0.100: Maryland lowered its drinking age to 18 for beer and wine in 1974, and kept 21 for liquor.
  Fact 0.101: Maryland raised its drinking age to 21 in 1982, with a grandfather clause.
  Fact 0.102: Maryland's 21st-century drinking age is 21, and anyone can drink when parents, spouses or teachers are present.
  Fact 0.103: Massachusetts's post-Prohibition drinking age was 21.
  Fact 0.104: Massachusetts lowered its drinking age to 18 in 1973.
  Fact 0.105: Massachusetts raised its drinking age to 20 on April 16, 1979.
  Fact 0.106: Massachusetts raised its drinking age to 21 in 1985.
  Fact 0.107: Massachusetts's 21st-century drinking age is 21.
  Fact 0.108: Michigan's post-Prohibition drinking age was 21.
  Fact 0.109: Michigan lowered its drinking age to 18 on January 1, 1972.
  Fact 0.110: Michigan raised its drinking age to 19 on December 3, 1978.
  Fact 0.111: Michigan raised its drinking age to 21 on December 21, 1978, 18 days later.
  Fact 0.112: Michigan was the first state to raise its age to 21 after lowering it.
  Fact 0.113: Michigan's 21st-century drinking age is 21.
  Fact 0.114: Minnesota's post-Prohibition drinking age was 21.
  Fact 0.115: Minnesota lowered its drinking age to 18 on June 1, 1973.
  Fact 0.116: Minnesota raised its drinking age to 19 on September 1, 1976.
  Fact 0.117: Minnesota was the first state to raise its drinking age after lowering it.
  Fact 0.118: Minnesota raised its drinking age to 21 in 1986.
  Fact 0.119: Minnesota's 21st-century drinking age is 21.
  Fact 0.120: Mississippi set a drinking age of 18 for beer and wine and 21 for liquor in 1966, when alcohol was legalized.
  Fact 0.121: Mississippi's drinking age was 18 in the 1970s.
  Fact 0.122: Mississippi raised its drinking age to 21 in 1986.
  Fact 0.123: Mississippi's 21st-century drinking age is 21.
  Fact 0.124: Missouri's drinking age was 21 from 1945 onward.
  Fact 0.125: Montana's post-Prohibition drinking age was 21.
  Fact 0.126: Montana lowered its drinking age to 19 in 1971.
  Fact 0.127: Montana lowered its drinking age to 18 in 1973.
  Fact 0.128: Montana raised its drinking age to 19 in 1979.
  Fact 0.129: Montana raised its drinking age to 21 in 1987.
  Fact 0.130: Montana's 21st-century drinking age is 21.
  Fact 0.131: Nebraska's post-Prohibition drinking age was 21.
  Fact 0.132: Nebraska lowered its drinking age to 20 in 1969.
  Fact 0.133: Nebraska lowered its drinking age to 19 in 1972.
  Fact 0.134: Nebraska raised its drinking age to 20 in 1980.
  Fact 0.135: Nebraska raised its drinking age to 21 in 1985.
  Fact 0.136: Nebraska's 21st-century drinking age is 21.
  Fact 0.137: Nevada's drinking age was 21 from 1933 onward.
  Fact 0.138: New Hampshire's post-Prohibition drinking age was 21.
  Fact 0.139: New Hampshire lowered its drinking age to 18 in 1973.
  Fact 0.140: New Hampshire raised its drinking age to 20 in 1979.
  Fact 0.141: New Hampshire raised its drinking age to 21 in 1985.
  Fact 0.142: New Hampshire's 21st-century drinking age is 21.
  Fact 0.143: New Jersey had no drinking age before 1880.
  Fact 0.144: New Jersey set a drinking age of 18 after 1880, with penalties only against businesses.
  Fact 0.145: New Jersey's post-Prohibition drinking age was 21.
  Fact 0.146: New Jersey lowered its drinking age to 18 in 1973.
  Fact 0.147: New Jersey raised its drinking age to 19 in 1980.
  Fact 0.148: New Jersey raised its drinking age to 21 in 1983.
  Fact 0.149: New Jersey's 21st-century drinking age is 21.
  Fact 0.150: New Mexico's drinking age was 21 from 1934 onward.
  Fact 0.151: New York's pre-Prohibition drinking age was 21.
  Fact 0.152: New York's post-Prohibition drinking age was 18.
  Fact 0.153: New York's drinking age was 18 in the 1970s.
  Fact 0.154: New York raised its drinking age to 19 in December 1982.
  Fact 0.155: New York raised its drinking age to 21 in December 1985.
  Fact 0.156: New York's 21st-century drinking age is 21.
  Fact 0.157: North Carolina's post-Prohibition drinking age was 18.
  Fact 0.158: North Carolina's 1970s drinking age was 18 for beer and wine and 21 for liquor.
  Fact 0.159: North Carolina's 1983 drinking age was 19 for beer and wine and 21 for liquor.
  Fact 0.160: North Carolina raised its drinking age to 21 on September 1, 1986.
  Fact 0.161: North Carolina's 21st-century drinking age is 21.
  Fact 0.162: North Dakota's drinking age was 21 from 1936 onward.
  Fact 0.163: Ohio's pre-Prohibition drinking age is unknown.
  Fact 0.164: Ohio set its post-Prohibition drinking age in 1935 at 18 for 3.2% ABW beer and 21 for beer stronger than 3.2% ABW, wine and liquor.
  Fact 0.165: Ohio's 1970s drinking age was 18 for 3.2% ABW beer and 21 for beer stronger than 3.2% ABW, wine and liquor.
  Fact 0.166: In 1982 Ohio set the age at 19 for 3.2% ABW beer and 21 for beer stronger than 3.2% ABW, wine and liquor.
  Fact 0.167: A second 1982 Ohio entry set the age at 19 for beer and 21 for wine and liquor.
  Fact 0.168: Ohio raised its drinking age to 21 in 1988.
  Fact 0.169: Ohio's 21st-century drinking age is 21, and anyone can drink with parents or a legal-age spouse.
  Fact 0.170: Oklahoma's drinking age was 21 for men and 18 for women for 3.2% ABW beer, and 21 for beer stronger than 3.2% ABW, wine and liquor.
  Fact 0.171: Oklahoma lowered its drinking age to 18 for 3.2% ABW beer for both sexes in December 1976.
  Fact 0.172: Oklahoma was the last state to lower the drinking age.
  Fact 0.173: Oklahoma raised its drinking age to 21 in 1983.
  Fact 0.174: Oklahoma's 21st-century drinking age is 21.
  Fact 0.175: Oregon's drinking age was 21 from 1933 onward.
  Fact 0.176: Pennsylvania's drinking age was 21 from 1935 onward.
  Fact 0.177: Rhode Island's post-Prohibition drinking age was 21.
  Fact 0.178: Rhode Island lowered its drinking age to 18 in 1970.
  Fact 0.179: Rhode Island raised its drinking age to 19 in 1980.
  Fact 0.180: Rhode Island raised its drinking age to 20 in 1981.
  Fact 0.181: Rhode Island raised its drinking age to 21 in 1984.
  Fact 0.182: Rhode Island's 21st-century drinking age is 21.
  Fact 0.183: South Carolina's pre-Prohibition drinking age was 21.
  Fact 0.184: South Carolina set its post-Prohibition drinking age in 1935 at 18 for beer and wine and 21 for liquor.
  Fact 0.185: South Carolina's 1970s drinking age was 18 for beer and wine and 21 for liquor.
  Fact 0.186: South Carolina's 1984 drinking age was 19 for beer and wine and 21 for liquor.
  Fact 0.187: South Carolina raised its drinking age to 21 in 1986.
  Fact 0.188: South Carolina's 21st-century drinking age is 21.
  Fact 0.189: South Dakota's drinking age was 21 for all alcohol in 1934.
  Fact 0.190: South Dakota lowered its drinking age to 18 for 3.2% beer in 1939 and raised it to 19 for 3.2% beer.
  Fact 0.191: South Dakota's 1972 drinking age was 18 for beer and wine and 21 for liquor.
  Fact 0.192: South Dakota's 1984 drinking age was 19 for beer and wine and 21 for liquor.
  Fact 0.193: South Dakota raised its drinking age to 21 in 1988.
  Fact 0.194: South Dakota's 21st-century drinking age is 21.
  Fact 0.195: Tennessee had no drinking age before 1881.
  Fact 0.196: Tennessee set its drinking age at 21 in 1881.
  Fact 0.197: Tennessee's post-Prohibition drinking age was 21.
  Fact 0.198: Tennessee lowered its drinking age to 18 in 1964.
  Fact 0.199: Tennessee raised its drinking age to 19 in 1979.
  Fact 0.200: Tennessee raised its drinking age to 21 in 1984.
  Fact 0.201: Tennessee's 21st-century drinking age is 21.
  Fact 0.202: Texas's drinking age was 21 before 1909, with none with written permission of parent or guardian.
  Fact 0.203: Texas's drinking age was 21 after 1909 and after Prohibition.
  Fact 0.204: Texas lowered its drinking age to 18 in 1973.
  Fact 0.205: Texas raised its drinking age to 19 in 1981.
  Fact 0.206: Texas raised its drinking age to 21 on September 1, 1986.
  Fact 0.207: Texas's 21st-century drinking age is 21.
  Fact 0.208: Utah's drinking age was 21 from 1935 onward.
  Fact 0.209: Vermont's post-Prohibition drinking age was 21.
  Fact 0.210: Vermont lowered its drinking age to 18 in 1971.
  Fact 0.211: Vermont raised its drinking age to 21 in 1986.
  Fact 0.212: Vermont's 21st-century drinking age is 21.
  Fact 0.213: Virginia's post-Prohibition drinking age was 21.
  Fact 0.214: Virginia lowered its drinking age to 18 for beer and wine in 1974, while 21 for liquor remained.
  Fact 0.215: Virginia raised its drinking age to 19 for off-premises consumption in 1981.
  Fact 0.216: Virginia raised its drinking age to 19 for all beer in 1983.
  Fact 0.217: Virginia raised its drinking age to 21 in 1985.
  Fact 0.218: Virginia's 21st-century drinking age is 21.
  Fact 0.219: Washington's drinking age has been the age of majority since 1877.
  Fact 0.220: Washington's drinking age was 21 from after Prohibition onward.
  Fact 0.221: West Virginia's post-Prohibition drinking age was 18 for beer and wine and 21 for liquor.
  Fact 0.222: West Virginia lowered its drinking age to 18 for all alcohol in 1972.
  Fact 0.223: West Virginia raised its drinking age to 19 in 1983, and to 21 for non-residents.
  Fact 0.224: West Virginia raised its drinking age to 21 in 1986.
  Fact 0.225: West Virginia's 21st-century drinking age is 21.
  Fact 0.226: Wisconsin's drinking age was 18 for wine and liquor, with none for beer, from 1839 to 1866.
  Fact 0.227: Wisconsin set its drinking age at 21 for all alcohol after 1866.
  Fact 0.228: Wisconsin's post-Prohibition drinking age was 18 for beer and 21 for wine and spirits.
  Fact 0.229: In 1957 Wisconsin set a 21 limit for residents of bordering states with an age limit of 21.
  Fact 0.230: In 1963 Wisconsin raised its age to 21 for off-premises beer sales, while it remained 18 for on-premises.
  Fact 0.231: Wisconsin lowered its drinking age to 18 for all alcohol in March 1972.
  Fact 0.232: Wisconsin lifted the border state restriction in 1977.
  Fact 0.233: Wisconsin raised its drinking age to 19 in 1984.
  Fact 0.234: Wisconsin raised its drinking age to 21 on September 1, 1986.
  Fact 0.235: Wisconsin's 21st-century drinking age is 21, and anyone can drink when parents, spouses or teachers are present.
  Fact 0.236: Wyoming's post-Prohibition drinking age was 21.
  Fact 0.237: Wyoming lowered its drinking age to 19 in 1973.
  Fact 0.238: Wyoming raised its drinking age to 21 on July 1, 1988.
  Fact 0.239: Wyoming was the last state to raise its de jure drinking age to 21.
  Fact 0.240: Wyoming's 21st-century drinking age is 21.
  Fact 0.241: American Samoa's listed drinking age is 21.
  Fact 0.242: The Northern Mariana Islands' listed drinking age is 21.
  Fact 0.243: Puerto Rico's post-Prohibition drinking age was 18.
  Fact 0.244: Puerto Rico's 1970s drinking age was 18.
  Fact 0.245: Puerto Rico kept its drinking age at 18 despite a 10% highway funding penalty under the Drinking Age Act.
  Fact 0.246: Puerto Rico's 21st-century drinking age is 18.
  Fact 0.247: The Virgin Islands' drinking age was 18 in the 1980s and in the 21st century.
  Fact 0.248: Guam's listed drinking age includes 21 and 18.
  Fact 0.249: Guam's 1980s and 21st-century drinking age was 18.
  Fact 0.250: Guam raised its drinking age to 21 in 2010.
Sentence 1: From 1976 to 1983 , several states voluntarily raised their purchase ages to 19 ( or , less commonly , 20 or 21 ) , in part to combat drunk driving fatalities . In 1984 , Congress passed the National Minimum Drinking Age Act , which required states to raise their ages for purchase and public possession to 21 by October 1986 or lose 10 % of their federal highway funds . By mid-1988 , all 50 states and the District of Columbia had raised their purchase ages to 21 ( but not Puerto Rico , Guam , or the Virgin Islands , see Additional Notes below ) . South Dakota and Wyoming were the final two states to comply with the age 21 mandate . The current drinking age of 21 remains a point of contention among many Americans , because of it being higher than the age of majority ( 18 in most states ) and higher than the drinking ages of most other countries . The National Minimum Drinking Age Act is also seen as a congressional sidestep of the tenth amendment . Although debates have not been highly publicized , a few states have proposed legislation to lower their drinking age , while Guam has raised its drinking age to 21 in July 2010 .
  Fact 1.0: From 1976 to 1983, several states voluntarily raised their purchase ages to 19 (or, less commonly, 20 or 21).
  Fact 1.1: The voluntary purchase-age increases from 1976 to 1983 were in part to combat drunk driving fatalities.
  Fact 1.2: In 1984 Congress passed the National Minimum Drinking Age Act.
  Fact 1.3: The National Minimum Drinking Age Act required states to raise their ages for purchase and public possession to 21 by October 1986 or lose 10% of their federal highway funds.
  Fact 1.4: By mid-1988, all 50 states and the District of Columbia had raised their purchase ages to 21.
  Fact 1.5: Puerto Rico, Guam, and the Virgin Islands had not raised their purchase ages to 21 by mid-1988.
  Fact 1.6: South Dakota and Wyoming were the final two states to comply with the age 21 mandate.
  Fact 1.7: The current drinking age of 21 remains a point of contention among many Americans.
  Fact 1.8: The current drinking age of 21 is higher than the age of majority, which is 18 in most states.
  Fact 1.9: The current drinking age of 21 is higher than the drinking ages of most other countries.
  Fact 1.10: The National Minimum Drinking Age Act is seen as a congressional sidestep of the tenth amendment.
  Fact 1.11: Debates over the drinking age have not been highly publicized.
  Fact 1.12: A few states have proposed legislation to lower their drinking age.
  Fact 1.13: Guam raised its drinking age to 21 in July 2010.

Retrieved items:
<item 1>
U.S. history of alcohol minimum purchase age by state
21 1974 : Lowered to 18 : for beer and wine ( 21 : for liquor remained ) 1981 : Raised to 19 for off - premises consumption 1983 : Raised to 19 ( all beer )
1985 : Raised to 21 21 Washington `` Age of majority '' since 1877 21 21 21 21 West Virginia ? 18 : for beer and wine 21 : for liquor 1972 : Lowered to 18 ( all
) 1983 : Raised to 19 ( & 21 for non-residents ) 1986 : Raised to 21 21 Wisconsin 1839 -- 1866 : 18 : for wine and liquor None : for beer Post-1866 : 21 ( all )
18 : for beer 21 : for wine and spirits 1957 : 21 : for residents of bordering states with age limit 21 . 1963 : Raised to 21 ( off - premises beer sales ,
remained 18 for on - premises ) .
</item 1>

<item 2>
U.S. history of alcohol minimum purchase age by state
18 : for beer and wine 21 : for liquor 18 : for beer and wine 21 : for liquor 1986 : Raised to 21 with grandfather clause . 21 Florida ? 18 1980 : 19 1985 :
Raised to 21 with grandfather clause 21 Georgia ? 21 1972 : Lowered to 18 1984 : Raised to 19 1985 : Raised to 20 1986 : Raised to 21 21 Hawaii ? 21 1972 :
Lowered to 18 1986 : Raised to 21 21 Idaho ? 20 : for beer 21 : for wine and liquor 1972 : Lowered to 19 ( all ) 1987 : Raised to 21 in 1987 ( Apr 11 ) with
grandfather clause 21 Illinois 1872 : Age of majority ( none with written consent of parent ) pre 1961 : Age of majority 1961 : Raised to 21 1973 : Lowered to
19 : for beer and wine 21 : for liquor 1980 : Raised to 21 21 Indiana ?
</item 2>

<item 3>
U.S. history of alcohol minimum purchase age by state
21 1974 : Lowered to 18 : for beer and wine 21 : for liquor 1982 : Raised to 21 with grandfather clause 21 ( however anyone can drink when parents , spouses ,
teachers are present ) Massachusetts ? 21 1973 : Lowered to 18 1979 : Raised to 20 in 1979 ( Apr 16 ) Raised to 21 in 1985 21 Michigan ? 21 1972 : Lowered to 18
in 1972 ( Jan 1 ) 1978 : Raised to 19 in 1978 ( Dec 3 ) 1978 : Raised to 21 in 1978 ( Dec 21 ) , 18 days later . First state to raise age to 21 since it was
lowered 21 Minnesota ? 21 1973 : Lowered to 18 ( June 1 ) 1976 : Raised to 19 ( Sept 1 ) First state to raise age after lowering . 1986 : Raised to 21 21
Mississippi ?
</item 3>

<item 4>
U.S. history of alcohol minimum purchase age by state
1983 : Raised to 21 21 Oregon ? 1933 : 21 21 21 21 Pennsylvania ? 1935 : 21 21 21 21 Rhode Island ? 21 1970 : Lowered to 18 1980 : Raised to 19 1981 : Raised to
20 1984 : Raised to 21 21 South Carolina 21 1935 : 18 : for beer and wine 21 : for liquor 18 : for beer and wine 21 : for liquor 1984 : 19 : for beer and wine
21 : for liquor 1986 : Raised to 21 21 South Dakota ?
</item 4>

<item 5>
U.S. history of alcohol minimum purchase age by state
1934 : 21 21 21 21 New York 21 18 18 1982 : Raised to 19 in 1982 ( Dec ) 1985 : Raised to 21 in 1985 ( Dec ) 21 North Carolina ? 18 18 : for beer and wine 21 :
for liquor 1983 : 19 : for beer and wine 21 : for liquor 1986 : Raised to 21 in 1986 ( Sep 1 ) 21 North Dakota ? 1936 : 21 21 21 21 Ohio ?
</item 5>

<item 6>
U.S. history of alcohol minimum purchase age by state
21 1972 : Lowered to 19 1985 : Raised to 21 21 Arkansas Pre 1925 : None 1925 : 21 21 21 21 21 California Pre 1891 : Regulated by municipality / county ( common
age was 16 ) 1891 : 18 ( statewide ) 1933 : 21 21 21 21 ( except small amounts for religious ceremonies ) Colorado None 1945 : 18 : for beer 21 : for wine and
liquor 18 Raised to 21 in 1987 21 Connecticut ?
</item 6>

<item 7>
U.S. history of alcohol minimum purchase age by state
1972 : Lowered to 18 ( all ) in 1972 ( Mar ) 1977 : Border state restriction lifted . 1984 : Raised to 19 1986 : Raised to 21 ( September 1 , 1986 ) 21 (
however anyone can drink when parents , spouses , teachers are present ) Wyoming ? 21 1973 : Lowered to 19 1988 : Raised to 21 in 1988 ( Jul 1 ) Last state to
raise de jure drinking age to 21 21 American Samoa ? ? ? 21 ? 21 Northern Mariana Islands -- -- ? 21 ? 21 Puerto Rico ? 18 18 Kept at 18 despite 10 % highway
funding penalty under Drinking Age Act . 18 Virgin Islands ? ? ? 18 18 Guam ? 21 ? 18 18 Raised to 21 in 2010 Additional Notes ( edit ) Contrary to popular
belief , since the National Minimum Drinking Age Act of 1984 , not all states specifically prohibit minors ' and young adults ' consumption of alcohol in
private settings .
</item 7>

<item 8>
U.S. history of alcohol minimum purchase age by state
By mid-1988 , all 50 states and the District of Columbia had raised their purchase ages to 21 ( but not Puerto Rico , Guam , or the Virgin Islands , see
Additional Notes below ) . South Dakota and Wyoming were the final two states to comply with the age 21 mandate . The current drinking age of 21 remains a point
of contention among many Americans , because of it being higher than the age of majority ( 18 in most states ) and higher than the drinking ages of most other
countries . The National Minimum Drinking Age Act is also seen as a congressional sidestep of the tenth amendment . Although debates have not been highly
publicized , a few states have proposed legislation to lower their drinking age , while Guam has raised its drinking age to 21 in July 2010 .
</item 8>

<item 9>
U.S. history of alcohol minimum purchase age by state
1938 : 21 21 21 21 Louisiana ? As of 1948 : 18 18 1987 : Raised to 21 de jure in 1987 , but de facto age was still 18 until 1995 due to a sale loophole . 1995 :
De facto age raised to 21 in 1995 when loophole was closed . 1996 : In 1996 , briefly lowered by Louisiana Supreme Court to 18 until it reversed its decision ,
raising to 21 three months later . Other exceptions still remain .
</item 9>

<item 10>
U.S. history of alcohol minimum purchase age by state
Shortly after the ratification of the 21st amendment in December , most states set their purchase ages at 21 since that was the Voting age at the time . Most of
these limits remained constant until the early 1970s . From 1969 to 1976 , some 30 states lowered their purchase ages , generally to 18 . This was primarily
because the voting age was lowered from 21 to 18 in 1971 with the 26th amendment . A lot of states started to lower their minimum drinking age in response ,
most of this occurring in 1972 or 1973 . Twelve states kept their purchase ages at 21 since repeal of Prohibition and never changed them . From 1976 to 1983 ,
several states voluntarily raised their purchase ages to 19 ( or , less commonly , 20 or 21 ) , in part to combat drunk driving fatalities . In 1984 , Congress
passed the National Minimum Drinking Age Act , which required states to raise their ages for purchase and public possession to 21 by October 1986 or lose 10 %
of their federal highway funds .
</item 10>
=== END ITEM id=2add4c25e0447df5 ===

=== ITEM id=b840f8c7aa6c2c49 ===
Gold sentences and facts:
Sentence 0: Drawing is one of the major forms of expression within the visual arts . It is generally concerned with the marking of lines and areas of tone onto paper / other material , where the accurate representation of the visual world is expressed upon a plane surface . Traditional drawings were monochrome , or at least had little colour , while modern colored - pencil drawings may approach or cross a boundary between drawing and painting . In Western terminology , drawing is distinct from painting , even though similar media often are employed in both tasks . Dry media , normally associated with drawing , such as chalk , may be used in pastel paintings . Drawing may be done with a liquid medium , applied with brushes or pens . Similar supports likewise can serve both : painting generally involves the application of liquid paint onto prepared canvas or panels , but sometimes an underdrawing is drawn first on that same support .
  Fact 0.0: Drawing is one of the major forms of expression within the visual arts.
  Fact 0.1: Drawing is generally concerned with the marking of lines and areas of tone onto paper or other material.
  Fact 0.2: In drawing, the accurate representation of the visual world is expressed upon a plane surface.
  Fact 0.3: Traditional drawings were monochrome, or at least had little colour.
  Fact 0.4: Modern colored-pencil drawings may approach or cross a boundary between drawing and painting.
  Fact 0.5: In Western terminology, drawing is distinct from painting.
  Fact 0.6: Drawing and painting can employ similar media even though they are distinct tasks.
  Fact 0.7: Dry media normally associated with drawing, such as chalk, may be used in pastel paintings.
  Fact 0.8: Drawing may be done with a liquid medium applied with brushes or pens.
  Fact 0.9: Similar supports can serve both drawing and painting.
  Fact 0.10: Painting generally involves the application of liquid paint onto prepared canvas or panels.
  Fact 0.11: Sometimes an underdrawing is drawn first on the same support that is later painted.

Retrieved items:
<item 1>
Drawing uses atmospheric haze to emphasize foreground objects. Artistic composition arranges elements to communicate ideas and create visual harmony; lighting
and shadow shape mood and appearance, while artists consider both the figure and surrounding negative space. Background objects must be positioned convincingly.
Preparatory studies test details and techniques, though they may become artworks themselves. Drawing skill varies, and visual accuracy means depicting a
recognizable object in its specific setting without adding unseen or omitting visible details.
</item 1>

<item 2>
Drawing
The wide availability of drawing instruments makes drawing one of the most common artistic activities . In addition to its more artistic forms , drawing is
frequently used in commercial illustration , animation , architecture , engineering and technical drawing . A quick , freehand drawing , usually not intended as
a finished work , is sometimes called a sketch . An artist who practices or works in technical drawing may be called a drafter , draftsman or a draughtsman .
Contents ( hide ) 1 Overview 2 History 2.1 Drawing in communication 2.2 Drawing in manuscripts 2.3 Drawing in science 2.4 Drawing as artistic expression 3
Notable artists and draftsmen 4 Materials 5 Technique 6 Tone 7 Form and proportion 8 Perspective 9 Artistry 10 Process 11 Seven Ps of drawing 12 See also 13
References 14 External links Overview ( edit ) Galileo Galilei . Phases of the Moon . 1616 . Drawing is one of the major forms of expression within the visual
arts . It is generally concerned with the marking of lines and areas of tone onto paper / other material , where the accurate representation of the visual world
is expressed upon a plane surface .
</item 2>

<item 3>
Drawing
Traditional drawings were monochrome , or at least had little colour , while modern colored - pencil drawings may approach or cross a boundary between drawing
and painting . In Western terminology , drawing is distinct from painting , even though similar media often are employed in both tasks . Dry media , normally
associated with drawing , such as chalk , may be used in pastel paintings . Drawing may be done with a liquid medium , applied with brushes or pens . Similar
supports likewise can serve both : painting generally involves the application of liquid paint onto prepared canvas or panels , but sometimes an underdrawing is
drawn first on that same support . Drawing is often exploratory , with considerable emphasis on observation , problem - solving and composition . Drawing is
also regularly used in preparation for a painting , further obfuscating their distinction . Drawings created for these purposes are called studies Madame
Palmyre with Her Dog , 1897 . Henri de Toulouse - Lautrec There are several categories of drawing , including figure drawing , cartooning , doodling , and free
hand .
</item 3>

<item 4>
Drawing
At this point , drawing was commonly used as a tool for thought and investigation , acting as a study medium whilst artists were preparing for their final
pieces of work . The Renaissance brought about a great sophistication in drawing techniques , enabling artists to represent things more realistically than
before , and revealing an interest in geometry and philosophy . The invention of the first widely available form of photography led to a shift in the hierarchy
of the arts . Photography offered an alternative to drawing as a method for accurately representing visual phenomena , and traditional drawing practice was
given less emphasis as an essential skill for artists , particularly so in Western society . Example of a typical drawing by American artist and notable
draftsman , Jean - Michel Basquiat , Untitled ( Axe / Rene ) 1982 . Basquiat produced over 1500 drawings and pushed the boundaries of drawing in the late 20th -
century Notable artists and draftsmen ( edit ) Drawing became significant as an art form around the late 15th century , with artists and master engravers such
as Albrecht Dürer and Martin Schongauer ( c. 1448 - 1491 ) , the very first Northern engraver known by name .
</item 4>

<item 5>
Drawing evolved from a preparatory tool into a sophisticated art form, especially during the Renaissance, when geometry and realistic representation gained
importance. Photography later reduced drawing’s status as a necessary representational skill. Drawing traditions reflected local culture and patronage: Dutch
works often stood alone as finished pieces, while Italian art reflected Church influence and French Classicism. Modernism encouraged abstraction. Artists
including Dürer, Picasso, Warhol, and Basquiat challenged conventions; Basquiat made over 1,500
</item 5>

<item 6>
Drawing
Drawing - wikipedia Drawing Jump to : navigation , search For other uses , see Drawing ( disambiguation ) . Pen and wash lion by Rembrandt in the Louvre Drawing
is a form of visual art in which a person uses various drawing instruments to mark paper or another two - dimensional medium . Instruments include graphite
pencils , pen and ink , inked brushes , wax colored pencils , crayons , charcoal , chalk , pastels , various kinds of erasers , markers , styluses , various
metals ( such as silverpoint ) and electronic drawing . A drawing instrument releases a small amount of material onto a surface , leaving a visible mark . The
most common support for drawing is paper , although other materials , such as cardboard , plastic , leather , canvas , and board , may be used . Temporary
drawings may be made on a blackboard or whiteboard or indeed almost anything . The medium has been a popular and fundamental means of public expression
throughout human history . It is one of the simplest and most efficient means of communicating visual ideas . The wide availability of drawing instruments makes
drawing one of the most common artistic activities .
</item 6>

<item 7>
Drawing
There are also many drawing methods , such as line drawing , stippling , shading , the surrealist method of entopic graphomania ( in which dots are made at the
sites of impurities in a blank sheet of paper , and lines are then made between the dots ) , and tracing ( drawing on a translucent paper , such as tracing
paper , around the outline of preexisting shapes that show through the paper ) . A quick , unrefined drawing may be called a sketch . In fields outside art ,
technical drawings or plans of buildings , machinery , circuitry and other things are often called `` drawings '' even when they have been transferred to
another medium by printing . History ( edit ) Drawing in communication ( edit ) Drawing is one of the oldest forms of human expression , with evidence for its
existence preceding that of written communication . It is believed that drawing was used as a specialised form of communication before the invention of the
written language , demonstrated by the production of cave and rock paintings around 30,000 years ago ( Art of the Upper Paleolithic ) . These drawings , known
as pictograms , depicted objects and abstract concepts .
</item 7>

<item 8>
Drawing
Once these basic volumes have been assembled into a likeness , then the drawing can be refined into a more accurate and polished form . The lines of the
primitive volumes are removed and replaced by the final likeness . Drawing the underlying construction is a fundamental skill for representational art , and is
taught in many books and schools . Its correct application resolves most uncertainties about smaller details , and makes the final image look consistent . A
more refined art of figure drawing relies upon the artist possessing a deep understanding of anatomy and the human proportions . A trained artist is familiar
with the skeleton structure , joint location , muscle placement , tendon movement , and how the different parts work together during movement . This allows the
artist to render more natural poses that do not appear artificially stiff . The artist is also familiar with how the proportions vary depending on the age of
the subject , particularly when drawing a portrait . Two - point perspective drawing Perspective ( edit ) Linear perspective is a method of portraying objects
on a flat surface so that the dimensions shrink with distance . Each set of parallel , straight edges of any object , whether a building or a table , follows
lines that eventually converge at a vanishing point .
</item 8>

<item 9>
Drawing
Pen and ink drawings often use hatching -- groups of parallel lines . Cross-hatching uses hatching in two or more different directions to create a darker tone .
Broken hatching , or lines with intermittent breaks , form lighter tones -- and controlling the density of the breaks achieves a gradation of tone . Stippling
uses dots to produce tone , texture and shade . Different textures can be achieved depending on the method used to build tone . Drawings in dry media often use
similar techniques , though pencils and drawing sticks can achieve continuous variations in tone . Typically a drawing is filled in based on which hand the
artist favors . A right - handed artist draws from left to right to avoid smearing the image . Erasers can remove unwanted lines , lighten tones , and clean up
stray marks . In a sketch or outline drawing , lines drawn often follow the contour of the subject , creating depth by looking like shadows cast from a light in
the artist 's position . Sometimes the artist leaves a section of the image untouched while filling in the remainder .
</item 9>

<item 10>
Drawing is a visual art and a fundamental means of communication, made by marking paper or other surfaces with tools such as pencils, ink, charcoal, chalk,
pastels, and digital styluses. It ranges from monochrome sketches to colored work and is used in art, illustration, animation, architecture, engineering, and
science. Common forms include figure drawing, cartooning, and technical plans; methods include line work, stippling, shading, and tracing. Drawing predates
writing: prehistoric cave images
</item 10>
=== END ITEM id=b840f8c7aa6c2c49 ===

=== ITEM id=c6f73dc78e325129 ===
Gold sentences and facts:
Sentence 0: The official music video for the song was directed by P.R. Brown and produced by Steve Lamar . The video was filmed along the Californian coast at The Inn at Newport Ranch , a resort and cattle ranch to the north of San Francisco . It features Brown and his duet partner Lauren Alaina singing along the coast . It was released on May 14 , 2017 . An earlier lyric video was first released on May 4 , 2017 , and shows Brown and Alaina cruising down a road .
  Fact 0.0: The official music video for the song was directed by P.R. Brown.
  Fact 0.1: The official music video for the song was produced by Steve Lamar.
  Fact 0.2: The video was filmed along the Californian coast at The Inn at Newport Ranch.
  Fact 0.3: The Inn at Newport Ranch is a resort and cattle ranch.
  Fact 0.4: The Inn at Newport Ranch is north of San Francisco.
  Fact 0.5: The video features Brown and his duet partner Lauren Alaina singing along the coast.
  Fact 0.6: The video was released on May 14, 2017.
  Fact 0.7: An earlier lyric video was first released on May 4, 2017.
  Fact 0.8: The earlier lyric video shows Brown and Alaina cruising down a road.

Retrieved items:
<item 1>
What Ifs
Music Canada . Retrieved February 8 , 2018 . Preceded by `` Body Like a Back Road '' by Sam Hunt Billboard Hot Country Songs number - one single October 21 --
November 11 , 2017 Succeeded by `` When It Rains It Pours '' by Luke Combs Preceded by `` All the Pretty Girls '' by Kenny Chesney Billboard Country Airplay
number - one single October 28 , 2017 Kane Brown Studio albums Kane Brown ( 2016 ) EPs Closer ( 2015 ) Chapter 1 ( 2016 ) Singles `` Do n't Go City on Me '' ``
Used to Love You Sober '' `` What Ifs '' `` Heaven '' Lauren Alaina Studio albums Wildflower Road Less Traveled EPs Lauren Alaina Singles `` Like My Mother Does
'' `` Georgia Peaches '' `` Eighteen Inches '' `` Barefoot and Buckwild '' `` Next Boyfriend '' `` Road Less Traveled '' `` Doin ' Fine '' Other singles `` My
Grown Up Christmas List '' `` What Ifs '' Retrieved from `` https://en.wikipedia.org/w/index.
</item 1>

<item 2>
What Ifs
According to Brown , another song was going to be used in the album , however , he was asked so much about `` What Ifs '' on social media that it was then
decided that the song should be included in the album . Composition ( edit ) The song is in the key of G minor with a moderately fast tempo of approximately 126
beats per minute . It follows the chord progression Gm - B ♭ - F. Brown 's vocals range from F - B ♭ . Commercial performance ( edit ) The song sold 6,000
downloads in the US the week the album was released on December 2 , 2016 , allowing it to chart at number 37 on the US Billboard Hot Country Songs chart . The
song first entered the US Billboard Country Airplay chart for the week of March 4 , 2017 at number 60 . It reached No. 1 on Hot Country Songs in October 2017 ,
ending the record breaking 34 - week reign of Sam Hunt 's `` Body Like a Back Road '' . The song is the first No. 1 on the chart for both Brown and Alaina .
</item 2>

<item 3>
What Ifs
wikipedia.org/w/index.php?title=What_Ifs&oldid=838232791 '' Categories : 2016 songs 2017 singles Kane Brown songs Lauren Alaina songs RCA Records Nashville
singles Vocal duets Song recordings produced by Dann Huff Songs written by Jordan Schmidt Billboard Hot Country Songs number - one singles Hidden categories :
Music infoboxes with deprecated parameters Articles with hAudio microformats Singlechart usages for Canada Singlechart called without song Singlechart usages
for Billboardcanadacountry Singlechart usages for Billboardhot100 Singlechart usages for Billboardcountryairplay Singlechart usages for Billboardcountrysongs
Certification Table Entry usages for Canada Certification Table Entry usages for United States Talk Contents About Wikipedia Español Edit links This page was
last edited on 25 April 2018 , at 18 : 59 . About Wikipedia
</item 3>

<item 4>
Salton Sea
Transportation Water Index of articles Regions Antelope Valley Big Sur California Coast Ranges Cascade Range Central California Central Coast Central Valley
Channel Islands Coachella Valley Coastal California Conejo Valley Cucamonga Valley Death Valley East Bay ( SF Bay Area ) East County ( SD ) Eastern California
Emerald Triangle Gold Country Great Basin Greater San Bernardino Inland Empire Klamath Basin Lake Tahoe Greater Los Angeles Los Angeles Basin Lost Coast Mojave
Desert Mountain Empire North Bay ( SF ) North Coast North Coast ( SD ) Northern California Owens Valley Oxnard Plain Peninsular Ranges Pomona Valley Sacramento
Valley Salinas Valley San Fernando Valley San Francisco Bay Area San Francisco Peninsula San Gabriel Valley San Joaquin Valley Santa Clara Valley Santa Clara
River Valley Santa Clarita Valley Santa Ynez Valley Shasta Cascade Sierra Nevada Silicon Valley South Bay ( LA ) South Bay ( SD ) South Bay ( SF ) South Coast
Southern Border Region Southern California Transverse Ranges Tri-Valley Victor Valley Wine Country Metro regions Metropolitan Fresno Los Angeles metropolitan
area Greater Sacramento San Bernardino - Riverside metropolitan area San Francisco metropolitan area San Diego -- Tijuana Counties Alameda Alpine Amador Butte
Calaveras Colusa Contra Costa Del Norte El Dorado Fresno Glenn Humboldt Imperial Inyo Kern Kings Lake Lassen
</item 4>

<item 5>
What Ifs
The song is the first No. 1 on the chart for both Brown and Alaina . The following week , for charts dated October 28 , 2017 , `` What Ifs '' stayed at number
one , as well as topping the Country Airplay and Country Streaming Songs charts . Together with the release of the deluxe edition of the album which also
reached number on Top Country Albums , and another number one for a new track `` Heaven '' on digital sales , Brown became the first artist to have simultaneous
number ones on all five main country charts . The song was certified Platinum by the RIAA on October 2 , 2017 . It was the third most - consumed country song of
2017 , with 1,551,000 including streams ( 599,000 in pure sale ) sold in the year . It has sold 679,000 copies in the US as of March 2018 . Music video ( edit )
The official music video for the song was directed by P.R. Brown and produced by Steve Lamar . The video was filmed along the Californian coast at The Inn at
Newport Ranch , a resort and cattle ranch to the north of San Francisco .
</item 5>

<item 6>
What Ifs
Jump up ^ `` Kane Brown Chart History ( Canada Country ) '' . Billboard . Retrieved November 14 , 2017 . Jump up ^ `` Kane Brown Chart History ( Hot 100 ) '' .
Billboard . Retrieved October 24 , 2017 . Jump up ^ `` Kane Brown Chart History ( Country Airplay ) '' . Billboard . Retrieved October 17 , 2017 . Jump up ^ ``
Kane Brown Chart History ( Hot Country Songs ) '' . Billboard . Retrieved October 10 , 2017 . Jump up ^ `` Country Airplay Songs -- Year - End 2017 '' .
Billboard . Retrieved December 18 , 2017 . Jump up ^ `` Hot Country Songs -- Year - End 2017 '' . Billboard . Retrieved December 18 , 2017 . Jump up ^ `` Hot
100 Songs -- Year - End 2017 '' . Billboard . Retrieved December 12 , 2017 . Jump up ^ `` Canadian single certifications -- Kane Brown -- What Ifs '' . Music
Canada .
</item 6>

<item 7>
What Ifs
The song was released with the album through RCA Nashville and was serviced to radio as the third single on February 6 , 2017 . Contents ( hide ) 1 Background 2
Composition 3 Commercial performance 4 Music video 5 Chart performance 5.1 Weekly charts 5.2 Year - end charts 6 Certifications and sales 7 References
Background ( edit ) The song was written by Brown with Matthew McGinn and Jordan Schmidt . The song is about a couple where the question `` what if '' is asked
repeatedly about what would happen in their relationship . It features a duet with Lauren Alaina , a school friend from Lakeview - Fort Oglethorpe High School
when he was living in Fort Oglethorpe , Georgia . According to Brown , the song was initially not intended to be a duet , he said : `` Once we decided on adding
a woman to `` What Ifs , '' I knew that I wanted Lauren to sing it , and , of course , she killed it . ''
</item 7>

<item 8>
What Ifs
What Ifs - wikipedia What Ifs Jump to : navigation , search `` What Ifs '' Single by Kane Brown featuring Lauren Alaina from the album Kane Brown Released
February 6 , 2017 ( 2017 - 02 - 06 ) Format Digital download Genre Country Length 3 : 08 Label RCA Nashville Songwriter ( s ) Kane Brown Matthew McGinn Jordan
Schmidt Producer ( s ) Dann Huff Kane Brown singles chronology `` Thunder in the Rain '' ( 2016 ) `` What Ifs '' ( 2017 ) `` Heaven '' ( 2017 ) `` Thunder in
the Rain '' ( 2016 ) `` What Ifs '' ( 2017 ) `` Heaven '' ( 2017 ) Lauren Alaina singles chronology `` Road Less Traveled '' ( 2016 ) Road Less Traveled 2016 ``
What Ifs '' ( 2017 ) What Ifs2017 `` Doin ' Fine '' ( 2017 ) Doin ' Fine 2017 `` What Ifs '' is a song recorded by American country music singer Kane Brown
featuring Lauren Alaina for his self - titled debut album .
</item 8>

<item 9>
What Ifs
Jump up ^ Bjorke , Matt ( January 3 , 2018 ) . `` Top 30 Country Digital Singles Chart : January 3 , 2018 '' . Roughstock . Retrieved January 14 , 2017 . ^ Jump
up to : Bjorke , Matt ( March 6 , 2018 ) . `` The Top 30 Digital Country Singles : March 5 , 2018 '' . Roughstock . Retrieved March 11 , 2018 . Jump up ^ ``
Watch Kane Brown , Lauren Alaina 's Picturesque ' What Ifs ' Video '' . Rolling Stone . May 15 , 2017 . Jump up ^ Kimble , Lindsay ( May 4 , 2017 ) . `` From
Middle School Chorus to Nashville , Kane Brown and Lauren Alaina Team Up for Duet ' What Ifs ' '' . Jump up ^ `` Kane Brown Chart History ( Canadian Hot 100 )
'' . Billboard . Retrieved October 24 , 2017 . Jump up ^ `` Kane Brown Chart History ( Canada Country ) '' . Billboard .
</item 9>

<item 10>
What Ifs
It features Brown and his duet partner Lauren Alaina singing along the coast . It was released on May 14 , 2017 . An earlier lyric video was first released on
May 4 , 2017 , and shows Brown and Alaina cruising down a road . Chart performance ( edit ) Weekly charts ( edit ) Chart ( 2017 ) Peak position Canada (
Canadian Hot 100 ) 45 Canada Country ( Billboard ) US Billboard Hot 100 26 US Country Airplay ( Billboard ) US Hot Country Songs ( Billboard ) Year - end charts
( edit ) Chart ( 2017 ) Position US Billboard Country Airplay 7 US Billboard Hot Country Songs US Billboard Hot 100 72 Certifications and sales ( edit ) Region
Certification Certified units / Sales Canada ( Music Canada ) Platinum 80,000 United States ( RIAA ) Platinum 679,000 sales figures based on certification alone
shipments figures based on certification alone References ( edit ) Jump up ^ Dukes , Billy ( January 24 , 2017 ) . `` Kane Brown ( Feat . Lauren Alaina ) , '
What Ifs ' ( Listen ) '' . Taste of Country .
</item 10>
=== END ITEM id=c6f73dc78e325129 ===

=== ITEM id=446d9741f7fc4d5d ===
Gold sentences and facts:
Sentence 0: The College Football Playoff selection committee chose the semifinalists following the conclusion of the 2017 regular season . Alabama and Georgia advanced to the national championship after winning the semifinal games hosted by the Sugar Bowl and the Rose Bowl , respectively , on January 1 , 2018 . The championship game was played at Mercedes - Benz Stadium in Atlanta , Georgia on January 8 , 2018 .
  Fact 0.0: The College Football Playoff selection committee chose the semifinalists.
  Fact 0.1: The semifinalists were chosen following the conclusion of the 2017 regular season.
  Fact 0.2: Alabama advanced to the national championship.
  Fact 0.3: Georgia advanced to the national championship.
  Fact 0.4: Alabama won its semifinal game hosted by the Sugar Bowl on January 1, 2018.
  Fact 0.5: Georgia won its semifinal game hosted by the Rose Bowl on January 1, 2018.
  Fact 0.6: The championship game was played at Mercedes-Benz Stadium in Atlanta, Georgia.
  Fact 0.7: The championship game was played on January 8, 2018.
Sentence 1: The Mercedes - Benz Stadium in Atlanta , Georgia was announced as the host site for the fourth College Football National Championship on November 4 , 2015 .
  Fact 1.0: Mercedes-Benz Stadium in Atlanta, Georgia was announced as the host site for the fourth College Football National Championship.
  Fact 1.1: The host site announcement was made on November 4, 2015.

Retrieved items:
<item 1>
2018 College Football Playoff National Championship
Not an officially sanctioned NCAA championship Retrieved from ``
https://en.wikipedia.org/w/index.php?title=2018_College_Football_Playoff_National_Championship&oldid=822294017 '' Categories : 2017 -- 18 NCAA football bowl
games 2017 -- 18 College Football Playoff 2018 in sports in Georgia ( U.S. state ) Sports competitions in Atlanta American football in Georgia ( U.S. state )
2010s in Atlanta Alabama Crimson Tide football bowl games Georgia Bulldogs football bowl games January 2018 sports events in the United States Hidden categories
: Use mdy dates from January 2018 Talk Contents About Wikipedia Français Edit links This page was last edited on 25 January 2018 , at 15 : 15 . Text is
available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and
Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 1>

<item 2>
2018 College Football Playoff National Championship
Alabama and Georgia advanced to the national championship after winning the semifinal games hosted by the Sugar Bowl and the Rose Bowl , respectively , on
January 1 , 2018 . The championship game was played at Mercedes - Benz Stadium in Atlanta , Georgia on January 8 , 2018 . Contents ( hide ) 1 Background 2 2018
College Football Playoff 3 Teams 3.1 Alabama 3.2 Georgia 4 Game summary 4.1 Scoring summary 4.2 Statistics 5 Broadcasting 6 Dispute 7 See also 8 References 9
External links Background ( edit ) The Mercedes - Benz Stadium in Atlanta , Georgia was announced as the host site for the fourth College Football National
Championship on November 4 , 2015 .
</item 2>

<item 3>
Graph query row:
stadium: Mercedes-Benz Stadium
located_in: Atlanta
championship: 2018 College Football Playoff National Championship
relationship: LOCATED_IN
</item 3>

<item 4>
2018 College Football Playoff National Championship
champions Mythical national championship Plus - One system Power Five conferences Championship games for each season are played in January , while semifinal
games are identified by season year , alternating between December and January Links to related articles 2017 -- 18 NCAA football bowl game season New Mexico (
Dec. 16 ) Las Vegas ( Dec. 16 ) Camellia ( Dec. 16 ) Cure ( Dec. 16 ) New Orleans ( Dec. 16 ) Boca Raton ( Dec. 19 ) Frisco ( Dec. 20 ) Gasparilla ( Dec. 21 )
Famous Idaho Potato ( Dec. 22 ) Bahamas ( Dec. 22 ) Armed Forces ( Dec. 23 ) Birmingham ( Dec. 23 ) Dollar General ( Dec. 23 ) Hawaii ( Dec. 24 ) Quick Lane (
Dec. 26 ) Heart of Dallas ( Dec. 26 ) Cactus ( Dec. 26 ) Independence ( Dec. 27 ) Pinstripe ( Dec. 27 ) Texas ( Dec. 27 ) Foster Farms ( Dec. 27 ) Military (
Dec. 28 ) Camping World ( Dec. 28 ) Alamo ( Dec.
</item 4>

<item 5>
2018 College Football Playoff National Championship
28 ) Camping World ( Dec. 28 ) Alamo ( Dec. 28 ) Holiday ( Dec. 28 ) Belk ( Dec. 29 ) Cotton ( Dec. 29 ) Sun ( Dec. 29 ) Music City ( Dec. 29 ) Arizona ( Dec.
29 ) Fiesta ( Dec. 30 ) Liberty ( Dec. 30 ) Orange ( Dec. 30 ) TaxSlayer ( Dec. 30 ) Citrus ( Jan. 1 ) Outback ( Jan. 1 ) Peach ( Jan. 1 ) College Football
Playoff games : Rose ( semifinal ) ( Jan. 1 ) Sugar ( semifinal ) ( Jan. 1 ) National Championship Game ( Jan. 8 ) All - Star Games East -- West Shrine Game (
Jan. 20 ) NFLPA Collegiate Bowl ( Jan. 20 ) Senior Bowl ( Jan. 27 ) NCAA Football Championship Games Division I FBS 2007 * 2008 * 2009 * 2010 * 2011 * 2012 *
2013 * 2014 * 2015 * 2016 * 2017 * 2018 * 2019 *
</item 5>

<item 6>
2018 College Football Playoff National Championship
1982 1984 1985 1986 1988 1989 1990 1991 1992 1993 1994 1995 1997 1998 1999 2000 2001 2002 2003 2005 2006 2007 2008 2009 2011 2012 2013 2014 2015 2016 2017 2018
Division III 1973 1976 1977 1978 1979 1980 1981 1982 1984 1985 1986 1988 1989 1990 1991 1992 1993 1994 1995 1997 1998 1999 2000 2001 2002 2003 2005 2006 2007
2008 2009 2011 2012 2013 2014 2015 2016 2017 2018 Games played in January are denoted by an asterisk Alabama Crimson Tide bowl games 1926 Rose Bowl 1927 Rose
Bowl 1931 Rose Bowl
</item 6>

<item 7>
2018 College Football Playoff National Championship
Jump up ^ Alex Scarbrough ( November 4 , 2015 ) . `` Atlanta , Santa Clara and New Orleans land CFP title games for 2018 -- 20 '' . ESPN.com . Retrieved
November 4 , 2015 . ^ Jump up to : `` Selection Committee Rankings : Final Top 25 Rankings '' ( PDF ) . College Football Playoff . December 3 , 2017 . Retrieved
December 3 , 2017 . Jump up ^ `` Alabama 's Sugar Bowl win against Clemson sets up all - SEC final : Final score , stats , analysis '' . nola.com . January 1 ,
2018 . Retrieved January 2 , 2018 . Jump up ^ Emerson , Seth ( January 1 , 2018 ) . `` Georgia to play for national championship after Rose Bowl win in 2 OT ''
. ajc.com . Retrieved January 2 , 2018 . ^ Jump up to : Hoffman , Benjamin ; Drape , Joe ; Tracy , Marc ( January 8 , 2018 ) . `` National Championship Game :
Georgia vs.
</item 7>

<item 8>
2018 College Football Playoff National Championship
related to 2018 College Football Playoff National Championship at Wikimedia Commons College Football Playoff Championships Overview College Football Playoff
National Championship Games 2014 -- 15 2015 -- 16 2016 -- 17 2017 -- 18 2018 -- 19 2019 -- 20 2020 -- 21 Semifinal Bowls Rose 2014 ( Jan 2015 ) 2017 ( Jan 2018
) 2020 ( Jan 2021 ) Sugar 2014 ( Jan 2015 ) 2017 ( Jan 2018 ) 2020 ( Jan 2021 ) Orange 2015 ( Dec 2015 ) 2018 ( Dec 2018 ) Cotton 2015 ( Dec 2015 ) 2018 ( Dec
2018 ) Peach 2016 ( Dec 2016 ) 2019 ( Dec 2019 ) Fiesta 2016 ( Dec 2016 ) 2019 ( Dec 2019 ) Other List of College Football Playoff games New Year 's Six College
Football Playoff National Championship Trophy NCAA Division I FBS champions Mythical national championship Plus - One system Power Five conferences Championship
games for each season are played in
</item 8>

<item 9>
2018 College Football Playoff National Championship
2018 College football Playoff national championship - wikipedia 2018 College football Playoff national championship Jump to : navigation , search 2018 College
Football Playoff National Championship Alabama Crimson Tide Georgia Bulldogs ( 12 -- 1 ) ( 13 -- 1 ) SEC SEC 26 23 Head coach : Nick Saban Head coach : Kirby
Smart AP Coaches CFP AP Coaches CFP OT Total Alabama 0 0 10 10 6 26 Georgia 0 13 7 0 23 Date January 8 , 2018 Season 2017 Stadium Mercedes - Benz Stadium
Location Atlanta , Georgia MVP Offensive : # 13 QB Tua Tagovailoa , Fr . Alabama Defensive : # 94 DT Da'Ron Payne , Jr .
</item 9>

<item 10>
2018 College Football Playoff National Championship
2018 College football Playoff ( edit ) Semifinals Championship January 1 -- Sugar Bowl Mercedes - Benz Superdome , New Orleans 1 Clemson 6 4 Alabama 24 January
8 -- National Championship Mercedes - Benz Stadium , Atlanta 4 Alabama ( OT ) 26 January 1 -- Rose Bowl Rose Bowl , Pasadena 3 Georgia 23 2 Oklahoma 48 3
Georgia ( 2OT ) 54 This bracket : view talk edit The College Football Playoff ( CFP ) selected four teams to advance to the playoff : the Clemson Tigers ,
Oklahoma Sooners , Georgia Bulldogs , and Alabama Crimson Tide . Teams ( edit ) Alabama ( edit ) See also : 2017 Alabama Crimson Tide football team The Alabama
Crimson Tide defeated the Clemson Tigers in the 2018 Sugar Bowl by a score of 24 -- 6 to reach the championship game . The Tide had a 12 -- 1 season , becoming
the SEC West Division co-champions with the Auburn Tigers . The Crimson Tide were coached by Nick Saban .
</item 10>
=== END ITEM id=446d9741f7fc4d5d ===
