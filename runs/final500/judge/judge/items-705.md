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

=== ITEM id=3e6f20ed42e60043 ===
Gold sentences and facts:
Sentence 0: `` Fore ! '' , originally an Australian interjection , is used to warn anyone standing or moving in the flight of a golf ball . The mention of the term in an 1881 Australian Golf Museum indicates that the term was in use at least as early as that period .
  Fact 0.0: Fore is originally an Australian interjection.
  Fact 0.1: Fore is used to warn anyone standing or moving in the flight of a golf ball.
  Fact 0.2: An 1881 Australian Golf Museum mention of the term exists.
  Fact 0.3: The 1881 mention indicates that the term Fore was in use at least as early as 1881.

Retrieved items:
<item 1>
Orc (Middle-earth)
About the goblins of The Hobbit , Tolkien wrote : They are not based on direct experience of mine ; but owe , I suppose , a good deal to the goblin tradition
... especially as it appears in George MacDonald , except for the soft feet which I never believed in . Middle - earth terms for Orcs ( edit ) In the High -
elven tongue Quenya , the word for `` Orc '' was urco , plural urqui , meaning `` bogey '' , or `` bogeyman '' , that is , something that provokes fear . In the
Grey - elven tongue Sindarin , it was orch , plural yrch . In the Dwarven tongue Khuzdul , it was rukhs , plural rakhâs . In the language of the Drúedain or
Wild Men , it was gorgûn . In the Black Speech of Mordor , the equivalent was Uruk , as can be seen in Uruk - hai , `` Orc - folk '' . Orc itself is from
Rohirric and the Hobbit - language , which shared linguistic roots , but the term is clearly related to the older Elvish words .
</item 1>

<item 2>
Fore (golf)
Fore ( golf ) - Wikipedia Fore ( golf ) Jump to : navigation , search The tough rubber core of a golf ball makes it a hazard to others following a wayward shot
, despite its weight not exceeding 45.93 g ( 1.62 oz ) `` Fore ! '' , originally an Australian interjection , is used to warn anyone standing or moving in the
flight of a golf ball . The mention of the term in an 1881 Australian Golf Museum indicates that the term was in use at least as early as that period . It is
believed to come from the military `` beware before '' , which an artilleryman about to fire would yell alerting nearby infantrymen to drop to the ground to
avoid the shells overhead . ( Before may mean `` in front of ( the gun being fired ) '' ; fore may mean `` ( look ) ahead '' . ) Other possible origins include
the term being derived from the term `` fore - caddie '' , a caddie waiting down range from the golfer to find where the ball lands .
</item 2>

<item 3>
Fore (golf)
These caddies were often warned about oncoming golf balls by a shout of the term `` fore - caddie '' which was eventually shortened to just `` fore ! '' . The
Colonel Bogey March is based on the descending minor third which the original Colonel Bogey whistled instead of yelling Fore around 1914 . Notes ( edit ) Jump
up ^ SND : Fore ^ Jump up to : Why Do Golfers Yell `` Fore '' for Errant Shots ? retrieved June 15 , 2007 Jump up ^ Derivation 4 Fore ! and Caddie retrieved
June 15 , 2007 Jump up ^ The Windsor Magazine : An Illustrated Monthly for Men and Women Vol. 33 , no . 192 ( December 1910 ) References ( edit ) Scott , James
Sibbald David , The British Army : Its Origin , Progress , and Equipment , 1868 Windsor Magazine : An Illustrated Monthly for Men and Women Vol. 32 , no . 292 (
December 1910 ) This golf - related article is a stub . You can help Wikipedia by expanding it .
</item 3>

<item 4>
Forensic firearm examination analyzes firearms, bullets and cartridges recovered from crime scenes to link ammunition to a weapon and potentially a weapon to an
individual. Rifling, introduced around the 16th century, spins bullets for accuracy and leaves distinctive marks from the barrel. The first documented
successful case occurred in London in 1835: a Bow Street Runner matched a murder victim’s bullet to a mold in a suspect’s home. Further evidence led to the
suspect’s conviction.
</item 4>

<item 5>
The Wikipedia page “Fore (golf)” is categorized under golf terminology, Scottish English, and golf stubs. It was last edited on September 28, 2017, and includes
links to several other language versions.
</item 5>

<item 6>
Forensic firearm examination
Forensic firearm examination - wikipedia Forensic firearm examination Jump to : navigation , search Part of a series on Forensic science Physiological ( show )
Anthropology Biology Bloodstain pattern analysis Botany Dentistry DNA phenotyping DNA profiling Entomology Epidemiology Medicine Palynology Pathology Podiatry
Toxicology Social ( show ) Psychiatry Psychology Psychotherapy Social work Criminalistics ( hide ) Accounting Body identification Chemistry Facial
reconstruction Fingerprint analysis Firearm examination Footwear evidence Forensic arts Profiling Gloveprint analysis Palmprint analysis Questioned document
examination Vein matching Digital forensics ( show ) Computer exams Data analysis Database study Mobile devices Network analysis Photography Video analysis
Audio analysis Related disciplines ( show ) Electrical engineering Engineering Fire investigation Fire accelerant detection Fractography Linguistics Materials
engineering Polymer engineering Statistics Traffic collision reconstruction Related articles ( show ) Crime scene CSI effect Perry Mason syndrome Pollen
calendar Skid mark Trace evidence Use of DNA in forensic entomology Outline Forensic firearm examination is the forensic process of examining the
characteristics of firearms as well as any cartridges or bullets left behind at a crime scene . Specialists in this field are tasked with linking bullets and
cartridges to weapons and weapons to individuals .
</item 6>

<item 7>
The excerpts trace several songs and works: golf’s “fore” may come from “fore-caddie,” while Colonel Bogey’s whistle inspired a march; “Wade in the Water” is a
1901-published spiritual with disputed Underground Railroad meanings; and hymns by Spafford and Scriven express faith and comfort. They also note Mozart’s
Symphony No. 40 and Barrett Browning’s love sonnets. Most material concerns “God Save the King/Queen”: its uncertain
</item 7>

<item 8>
Golfers’ warning “fore!” may derive from “fore-caddie,” shouted to alert caddies to incoming balls. Around 1914, the original Colonel Bogey reportedly whistled
a descending minor third instead of yelling the warning; this melody inspired the Colonel Bogey March. The source article is a stub.
</item 8>

<item 9>
Fore (golf)
You can help Wikipedia by expanding it . Retrieved from `` https://en.wikipedia.org/w/index.php?title=Fore_(golf)&oldid=802796028 '' Categories : Golf
terminology Scottish English Golf stubs Hidden categories : All articles with specifically marked weasel - worded phrases Articles with specifically marked
weasel - worded phrases from January 2016 All stub articles Talk About Wikipedia Hrvatski Italiano Norsk Srpskohrvatski / српскохрватски Svenska Edit links This
page was last edited on 28 September 2017 , at 14 : 56 . About Wikipedia
</item 9>

<item 10>
“Fore!” is a golf warning shouted to alert people in or near the path of a wayward ball, which can injure others despite weighing no more than 45.93 g (1.62
oz). The term, originally an Australian interjection, is documented in 1881. Its origin is uncertain: it may derive from the military warning “beware before,”
or from “fore-caddie,” who waits downrange to locate a ball.
</item 10>
=== END ITEM id=3e6f20ed42e60043 ===

=== ITEM id=689e30a1b69874ee ===
Gold sentences and facts:
Sentence 0: After their plane crashes in the woods , Dr. Meredith Grey ( Ellen Pompeo ) , Dr. Lexie Grey ( Chyler Leigh ) , Dr. Cristina Yang ( Sandra Oh ) , Dr. Arizona Robbins ( Jessica Capshaw ) , Dr. Derek Shepherd ( Patrick Dempsey ) , and Dr. Mark Sloan ( Eric Dane ) desperately fight to stay alive . Meredith is relatively unscathed , while the rest have serious injuries : the pilot , Jerry ( James LeGros ) , has a major spine injury , and Yang dislocates her arm . Robbins ' femur is broken and sticking through the skin , Sloan has serious internal injuries ; though initially adrenaline keeps him on his feet . Shepherd is sucked out the side of the plane and awakens alone in the wood ; his mangled hand having been pushed through the door of the plane . However , none are in as bad shape as Lexie , who is crushed under a piece of the plane . While Meredith searches for Shepherd , Yang and Sloan try to move the debris off Lexie . Eventually , the two realize that they can not save her , so Sloan holds her hand while she dies , telling her that he loves her . As Sloan tells her of the life the two were meant to have together , Lexie dies with a smile on her face just as Meredith and Yang are approaching .
  Fact 0.0: After their plane crashes in the woods, six doctors desperately fight to stay alive.
  Fact 0.1: Dr. Meredith Grey is played by Ellen Pompeo.
  Fact 0.2: Dr. Lexie Grey is played by Chyler Leigh.
  Fact 0.3: Dr. Cristina Yang is played by Sandra Oh.
  Fact 0.4: Dr. Arizona Robbins is played by Jessica Capshaw.
  Fact 0.5: Dr. Derek Shepherd is played by Patrick Dempsey.
  Fact 0.6: Dr. Mark Sloan is played by Eric Dane.
  Fact 0.7: Meredith is relatively unscathed.
  Fact 0.8: The other doctors have serious injuries.
  Fact 0.9: The pilot, Jerry (played by James LeGros), has a major spine injury.
  Fact 0.10: Yang dislocates her arm.
  Fact 0.11: Robbins' femur is broken and sticking through the skin.
  Fact 0.12: Sloan has serious internal injuries.
  Fact 0.13: Initially, adrenaline keeps Sloan on his feet.
  Fact 0.14: Shepherd is sucked out the side of the plane and awakens alone in the woods.
  Fact 0.15: Shepherd's mangled hand was pushed through the door of the plane.
  Fact 0.16: No one is in as bad shape as Lexie.
  Fact 0.17: Lexie is crushed under a piece of the plane.
  Fact 0.18: While Meredith searches for Shepherd, Yang and Sloan try to move the debris off Lexie.
  Fact 0.19: Eventually, Yang and Sloan realize they cannot save Lexie.
  Fact 0.20: Sloan holds Lexie's hand while she dies.
  Fact 0.21: Sloan tells Lexie that he loves her.
  Fact 0.22: Sloan tells Lexie about the life the two were meant to have together.
  Fact 0.23: Lexie dies with a smile on her face.
  Fact 0.24: Meredith and Yang are approaching just as Lexie dies.
Sentence 1: `` Flight '' is the twenty - fourth and final episode of the eighth season of the American television medical drama Grey 's Anatomy , and the show 's 172nd episode overall . It was written by series creator Shonda Rhimes , and directed by Rob Corn . The episode was originally broadcast on the American Broadcasting Company ( ABC ) in the United States on May 17 , 2012 . In the episode , six doctors from Seattle Grace Mercy West Hospital who are victims of an aviation accident fight to stay alive , but Dr. Lexie Grey ( Chyler Leigh ) ultimately dies . Other storylines occur in Seattle where Dr. Richard Webber ( James Pickens , Jr . ) plans his annual dinner for the departing residents , Dr. Owen Hunt ( Kevin McKidd ) fires Dr. Teddy Altman ( Kim Raver ) , and Dr. Miranda Bailey ( Chandra Wilson ) gets engaged .
  Fact 1.0: 'Flight' is the twenty-fourth and final episode of the eighth season of the American television medical drama Grey's Anatomy.
  Fact 1.1: 'Flight' is the show's 172nd episode overall.
  Fact 1.2: 'Flight' was written by series creator Shonda Rhimes.
  Fact 1.3: 'Flight' was directed by Rob Corn.
  Fact 1.4: 'Flight' was originally broadcast on the American Broadcasting Company (ABC) in the United States on May 17, 2012.
  Fact 1.5: In the episode, six doctors from Seattle Grace Mercy West Hospital who are victims of an aviation accident fight to stay alive.
  Fact 1.6: In the episode, Dr. Lexie Grey (played by Chyler Leigh) ultimately dies.
  Fact 1.7: Other storylines occur in Seattle.
  Fact 1.8: Dr. Richard Webber (played by James Pickens, Jr.) plans his annual dinner for the departing residents.
  Fact 1.9: Dr. Owen Hunt (played by Kevin McKidd) fires Dr. Teddy Altman (played by Kim Raver).
  Fact 1.10: Dr. Miranda Bailey (played by Chandra Wilson) gets engaged.

Retrieved items:
<item 1>
Flight (Grey's Anatomy)
Grey 's Anatomy has once again managed to shock with its season finale . '' While she appreciated the `` realism and authenticity that Grey 's is known for '' ,
she found the episode was `` almost too much '' as it was `` extremely gory and difficult to watch , initially because of the grisly wounds '' but later because
of the `` heavy and emotional things that transpired '' . She thought Pompeo gave one of her best performances when her character learned that her sister was
dead . Digital Spy 's Ben Lee found Leigh 's and Dane 's performances `` phenomenal '' and added that he had probably never seen a better performance from Dane
. He described the moment the two actors shared as `` truly poignant '' . To him , Lexie 's death felt like a finale , which was thus too early . As for what
was happening in Seattle Grace , he thought it was `` a bit pointless '' and `` uninteresting '' except Altman 's departure , which he deemed `` the most
significant moment at the hospital '' . Entertainment Weekly 's Tanner Stransky commented of Lexie 's death : `` It was an intense death .
</item 1>

<item 2>
Flight (Grey's Anatomy)
Lexie 's death devastates Meredith , who is still desperately trying to find her husband . Eventually , she and Shepherd reunite and they fix his hand as best
as they can . Meanwhile , back at Seattle Grace Mercy West Hospital , no one is aware of what has happened to the other doctors . Dr. Richard Webber ( James
Pickens , Jr . ) prepares the annual dinner for the departing residents , which Dr. Alex Karev ( Justin Chambers ) , Dr. Jackson Avery ( Jesse Williams ) , and
Dr. April Kepner ( Sarah Drew ) are dreading . Avery makes the choice to take a job offer at Tulane Medical Center , and he and Kepner share a moment . Dr. Ben
Warren ( Jason George ) and Dr. Miranda Bailey ( Chandra Wilson ) decide to get married , even though Warren is going to start his surgical internship in Los
Angeles . After realizing Dr. Teddy Altman ( Kim Raver ) was offered a chief position at United States Army Medical Command ( MEDCOM ) and is refusing to leave
Seattle out of loyalty , Dr.
</item 2>

<item 3>
Flight (Grey's Anatomy)
Mark Sloan ) 's . `` Flight '' earned Rhimes an NAACP Image Award nomination and it was also nominated under several categories of Entertainment Weekly 's
finale awards . Upon its initial airing , the episode was viewed in the United States by 11.44 million people , garnered a 4.1 / 11 Nielsen rating / share in
the 18 -- 49 demographic , ranking fourth for the night in terms of viewership , and registering as Thursday 's highest - rated drama . Contents ( hide ) 1 Plot
2 Production 3 Reception 4 References 5 External links Plot ( edit ) After their plane crashes in the woods , Dr. Meredith Grey ( Ellen Pompeo ) , Dr. Lexie
Grey ( Chyler Leigh ) , Dr. Cristina Yang ( Sandra Oh ) , Dr. Arizona Robbins ( Jessica Capshaw ) , Dr. Derek Shepherd ( Patrick Dempsey ) , and Dr. Mark Sloan
( Eric Dane ) desperately fight to stay alive .
</item 3>

<item 4>
Flight (Grey's Anatomy)
`` ' Grey 's Anatomy ' recap : Major death in season finale ' Flight ' '' . Digital Spy . Nat Mags . Retrieved May 22 , 2012 . Jump up ^ Stransky , Tanner ( May
18 , 2012 ) . `` ' Grey 's Anatomy ' season finale recap : A Grey Goes Down '' . Entertainment Weekly . Time Inc . Retrieved May 22 , 2012 . Jump up ^ Bierly ,
Mandi ( May 29 , 2012 ) . `` Grey 's Anatomy , Chyler Leigh , ... TV Season Finale Awards : YOUR 2012 Winners ! Photo 9 of 20 '' . Entertainment Weekly . Time
Inc . Retrieved June 17 , 2012 . Jump up ^ Bierly , Mandi ( May 29 , 2012 ) . `` Grey 's Anatomy , Ellen Pompeo TV Season Finales TV Season Finale Awards : YOUR
2012 Winners ! Photo 12 of 20 '' . Entertainment Weekly . Time Inc . Retrieved June 18 , 2012 .
</item 4>

<item 5>
Flight (Grey's Anatomy)
In the episode , six doctors from Seattle Grace Mercy West Hospital who are victims of an aviation accident fight to stay alive , but Dr. Lexie Grey ( Chyler
Leigh ) ultimately dies . Other storylines occur in Seattle where Dr. Richard Webber ( James Pickens , Jr . ) plans his annual dinner for the departing
residents , Dr. Owen Hunt ( Kevin McKidd ) fires Dr. Teddy Altman ( Kim Raver ) , and Dr. Miranda Bailey ( Chandra Wilson ) gets engaged . The episode marked
Leigh 's and Raver 's final appearance to the series . Exterior filming of the accident took place at Big Bear Lake , California . Jason George reprised his
role as a guest star , whereas James LeGros made his first appearance . The episode opened to mixed reviews from television critics , with some criticizing the
death of Lexie , but praising Leigh 's performance , in addition to Ellen Pompeo ( Dr. Meredith Grey ) 's and Eric Dane ( Dr. Mark Sloan ) 's .
</item 5>

<item 6>
Flight (Grey's Anatomy)
Mark Sloan ( Eric Dane ) desperately fight to stay alive . Meredith is relatively unscathed , while the rest have serious injuries : the pilot , Jerry ( James
LeGros ) , has a major spine injury , and Yang dislocates her arm . Robbins ' femur is broken and sticking through the skin , Sloan has serious internal
injuries ; though initially adrenaline keeps him on his feet . Shepherd is sucked out the side of the plane and awakens alone in the wood ; his mangled hand
having been pushed through the door of the plane . However , none are in as bad shape as Lexie , who is crushed under a piece of the plane . While Meredith
searches for Shepherd , Yang and Sloan try to move the debris off Lexie . Eventually , the two realize that they can not save her , so Sloan holds her hand
while she dies , telling her that he loves her . As Sloan tells her of the life the two were meant to have together , Lexie dies with a smile on her face just
as Meredith and Yang are approaching . Lexie 's death devastates Meredith , who is still desperately trying to find her husband .
</item 6>

<item 7>
Flight (Grey's Anatomy)
Lexie 's death was also nominated under the `` Best ( Presumed ) Death '' category , while the plane crash 's aftermath was nominated as the `` Best
Non-romantic Cliffhanger '' , and the episode in entirety was nominated for the special award for `` Biggest Regret That I Did n't See It , I Just Heard or Read
About It '' . Entertainment Weekly later named the scene where Meredith is crying one of the best crying scenes of 2012 . In TVLine 's review of 2012 , Lexie 's
death was runner - up for `` Biggest Tearjerker '' . The episode is nominated at the NAACP Image Awards under the Outstanding Writing in a Dramatic Series
category for Rhimes . References ( edit ) Jump up ^ Goldberg , Lesley ( June 25 , 2012 ) . `` Emmys 2012 : ' Grey 's Anatomy 's ' Sandra Oh on Cristina 's
Emotional Season '' . The Hollywood Reporter . Prometheus Global Media . Retrieved July 21 , 2012 . Jump up ^ Writer : Rhimes , Shonda .
</item 7>

<item 8>
Flight (Grey's Anatomy)
Everybody was very accommodating -- the crew , cast . And I opted to stay underneath ( the wreckage ) for the most part over two days rather than trying to get
in and out . '' In regard to the episode , Rhimes commented before it originally aired that it was difficult to write , largely because of the death of a main
character . She compared it to writing the season six finale , by explaining that the former was `` more painful '' to write . After the episode aired , Rhimes
repeated in a tweet that it was hard for her to write the finale , adding : `` I did not enjoy it . It made me sick and it made me sad . '' Rhimes also
explained the departure of Leigh , whose character died after the plane crash , by saying that the two came to an agreement on the decision to kill Lexie ,
after extensive discussion . Speaking of Raver 's departure whose character left Seattle Grace for MEDCOM , Rhimes elaborated that Raver was offered a contract
renewal , but declined . Reception ( edit ) Chyler Leigh 's performance was deemed `` phenomenal '' .
</item 8>

<item 9>
Flight (Grey's Anatomy)
I mean , how awful was it to watch one of Grey 's longest - running characters pass away so quickly -- and rather unceremoniously ? I get that Ms. Rhimes had to
do what she had to do -- and every show needs to be shaken up once in a while -- but I do n't love that Lexie was the one to die . Could it have been someone
less important somehow ? I guess it would have been too obvious to do Kepner . And you probably just do n't kill off a hottie like Mark Sloan , right ? ''
Stransky also complained about Robbins ' screams at the beginning of the episode , but enjoyed Bailey 's story . In an Entertainment Weekly poll that judged all
the television season finales of the year , Lexie 's death was voted the `` Top Tissue Moment '' , while Robbins ' injured leg and Shepherd 's mangled hand were
voted the `` Most Disturbing Image '' . The ending of the episode was also considered as the `` Best Ending to an Otherwise So - So Season '' .
</item 9>

<item 10>
Flight (Grey's Anatomy)
Owen Hunt ( Kevin McKidd ) , the chief of surgery , fires her to free her from the hospital where her husband died . As the episode ends , Hunt picks up his
messages to discover the surgical team never made it to Boise . The residents , finally excited to celebrate at Webber 's dinner , are left waiting for their
stranded friends . The remaining crash survivors are left struggling to stay awake as their last match goes out . In the closing monologue , Meredith repeats
the opening to the speech that Webber gave in the pilot episode of the series . Production ( edit ) `` It was so hard . It was physically and emotionally
challenging because you 're constantly at extremely high stakes . The physicality for Cristina , she had to do a lot of things with one arm , and I had to do a
lot of things with one shoe . You 're running around the mountain with one shoe and one arm basically and doing things that way . And it 's like we 've had this
cushy time on stages and in the hospital . But being out in the open on the side of a mountain was exhilarating ; you can not beat location work .
</item 10>
=== END ITEM id=689e30a1b69874ee ===

=== ITEM id=14f920ac4ce144e3 ===
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
*Downton Abbey* is a British period drama created by Julian Fellowes and co-produced by Carnival Films and Masterpiece. It premiered on ITV in 2010 and on PBS
in 2011. The series ran for six seasons and 52 episodes, concluding with a Christmas special broadcast in the UK on 25 December 2015. The article lists the
episodes and provides series-overview information.
</item 2>

<item 3>
The list covers *Downton Abbey* episodes from 2011–2015. Series 2 episodes 4–8 aired weekly from 9 October to 6 November 2011, written by Julian Fellowes and
directed by Brian Kelly, Andy Goddard or James Strong; reported audiences ranged from 11.30 to 12.45 million. Series 3–6 each had eight episodes plus a
Christmas special, airing September–November and on 25 December, with listed
</item 3>

<item 4>
List of Downton Abbey episodes
Christmas special No . overall Title Directed by Written by Original air date UK viewers ( millions ) 52 `` The Finale '' Michael Engler Julian Fellowes 25
December 2015 ( 2015 - 12 - 25 ) 11.22 29 September -- 31 December 1925 : Carson fumbles serving dinner when his hands begin shaking , suffering from an
unspecified hereditary disease his family called ' the palsy ' . Thomas secures a job with an elderly couple with a staff of only three , but is unhappy there .
Edith offers to expand Spratt 's magazine column to a whole page ; Denker discovers Spratt 's secret , and tries to get him dismissed by revealing it to the
Dowager Countess , but she finds the column jolly amusing and keeps him on . Molesley leaves service when the Headmaster offers him a teaching position , plus a
cottage ; upon parting , he and Baxter finally express their mutual romantic affection . Lord Merton tells Isobel he suffers from pernicious anaemia , but still
wants to marry her .
</item 4>

<item 5>
List of Downton Abbey episodes
With the future of Downton hanging in the balance , Mary and the Dowager Countess conspire to show Downton at its grandest to persuade Mrs Levinson to give
Downton more money and save them from financial ruin . Their attempts are impeded by the oven breaking down , giving Mrs Levinson a chance to take over running
the party herself in radical fashion . She reveals , when finally asked for help , that though she will `` entertain them in New York '' she can not provide
them with money , as her own late husband tied it up in the estate , much as Violet 's did . She tells them they have to adapt to the changing world , and
states her intent to return to America . Meanwhile downstairs , O'Brien and Thomas wage war against one another due to Alfred 's recent promotion to valet to
Matthew , whilst Mrs Levinson 's maid takes a shine to Alfred . Mrs Hughes faces a cancer scare that she shares only with Mrs Patmore and Doctor Clarkson , and
Isobel Crawley finds that Ethel has become a prostitute .
</item 5>

<item 6>
List of Downton Abbey episodes
overall No. in series Title Directed by Written by Original air date UK viewers ( millions ) 26 `` Episode One '' David Evans Julian Fellowes 22 September 2013
( 2013 - 09 - 22 ) 11.96 February 1922 . Six months on , a distraught Lady Mary mourns Matthew , and feels little motherly love for baby George , now six months
old . Lord Grantham and Branson dispute the management of the estate and struggle to pay death duties . Miss O'Brien suddenly leaves to serve Lady Flintshire ,
on her way to India . Needing a maid in a hurry , Cora hires former maid Edna , who has trained as a lady 's maid , on the strength of the reference which Tom
persuaded Mrs Hughes to give her ; Carson and Mrs Hughes see trouble ahead , and warn Tom to keep his distance . Miss West , the nanny of George and Sybbie ,
Mary 's son and Tom 's daughter , respectively , acts insolently with Thomas , and the underbutler lets Cora know of her tendency to leave them to their own
devices .
</item 6>

<item 7>
The references document *Downton Abbey* episode listings, BARB weekly audience rankings for ITV and its HD/+1 channels in 2011–2013, and ratings methodology
(seven-day consolidated data for series 1–5; 28-day data for 2014–15 Christmas specials). A 2015 Metro report said the show would end after its sixth series.
</item 7>

<item 8>
List of Downton Abbey episodes
Series 6 No . overall No. in series Title Directed by Written by Original air date UK viewers ( millions ) 44 `` Episode One '' Minkie Spiro Julian Fellowes 20
September 2015 ( 2015 - 09 - 20 ) 10.32 April 1925 . Big changes are afoot . Robert anticipates having to cut costs and reduce staff , worrying Carson . Mrs
Hughes , nervous she will be embarrassed when performing her wifely duties after marrying Carson , asks Mrs Patmore to sound him out ; after much embarrassment
, Carson says he expects them to have a full marriage , which reassures Mrs Hughes . The local hospital is threatened with a take - over by the larger County
Hospital ; the board is split , with Violet objecting , and Isobel approving in the name of efficiency . Rita Bevan , a former chambermaid at the hotel where
Mary and Lord Gillingham stayed , attempts to blackmail Mary by exposing their tryst .
</item 8>

<item 9>
*Downton Abbey*, Julian Fellowes’s six-season period drama (52 episodes), aired on ITV from 2010 to 2015 and on PBS from 2011. The material outlines episode
plots, production, renewals and BARB ratings across its series and Christmas specials; audiences often exceeded 10 million in the UK, while PBS’s Season 3
premiere drew a record 7.9 million. Series 5 began in 1924 amid political change and family troubles. *The
</item 9>

<item 10>
List of Downton Abbey episodes
Molesley and Baxter take a photo of Bates to different pubs in York , and find the one where he had lunch on the day in question , proving he was not in London
when Green was killed . The witness who identified Anna becomes uncertain , and she is released on bail . A big Christmas party is held for the Downton
household . It also serves as a goodbye party for Tom , who has found work as a writer in America and will be taking his daughter Sybbie with him . Bates
returns and enjoys a tender reunion with Anna . During a private break , Violet confesses to Isobel that she and Prince Kuragan had attempted to run away
together . Series 6 ( 2015 ) ( edit ) The sixth series premiered on 20 September 2015 in the UK and on 3 January 2016 in the United States . This series was the
last , the 2015 Christmas special titled `` The Finale '' . Due to increasingly popular ratings worldwide , a film is said to be in production . Series 6 No .
overall No.
</item 10>
=== END ITEM id=14f920ac4ce144e3 ===

=== ITEM id=85adde773634f6fa ===
Gold sentences and facts:
Sentence 0: A noteworthy example of a trabeated system is in Volubilis , from the Roman era , where one side of the Decumanus Maximus is lined with trabeated elements , while the opposite side of the roadway is designed in arched style .
  Fact 0.0: A noteworthy example of a trabeated system is in Volubilis.
  Fact 0.1: The Volubilis trabeated example is from the Roman era.
  Fact 0.2: In Volubilis, one side of the Decumanus Maximus is lined with trabeated elements.
  Fact 0.3: In Volubilis, the opposite side of the roadway is designed in arched style.

Retrieved items:
<item 1>
Post and lintel
The trabeated system is a fundamental principle of Neolithic architecture , Ancient Indian architecture , Ancient Greek architecture and Ancient Egyptian
architecture . Other trabeated styles are the Persian , Lycian , Japanese , traditional Chinese , and ancient Chinese architecture , especially in northern
China , and nearly all the Indian styles . The traditions are represented in North and Central America by Mayan architecture , and in South America by Inca
architecture . In all or most of these traditions , certainly in Greece and India , the earliest versions developed using wood , which were later translated
into stone for larger and grander buildings . Timber framing , also using trusses , remains common for smaller buildings such as houses to the modern day . The
biggest disadvantage to a post and lintel construction is the limited weight that can be held up , and the small distances required between the posts . Ancient
Roman architecture 's development of the arch allowed for much larger structures to be constructed . The arcuated system spreads larger loads more effectively ,
and replaced the post and lintel trabeated system in most larger buildings and structures , until the introduction of steel girder beams in the industrial era .
</item 1>

<item 2>
Post and lintel
Engineering ( edit ) Post and lintel construction is one of four ancient structural methods of building , the others being the corbel , arch - and - vault , and
truss . There are two main force vectors acting upon the post and lintel system : weight carrying compression at the joint between lintel and post , and tension
induced by deformation of self - weight and the load above between the posts . The two posts are under compression from the weight of the lintel ( or beam )
above . The lintel will deform by sagging in the middle because the underside is under tension and the topside is under compression . See also ( edit )
Wikimedia Commons has media related to Post and Beam and Lintels .
</item 2>

<item 3>
Florence Cathedral
The rails were connected end - to - end with lead - glazed iron splices . The cross ties and rails were notched together and then covered with the bricks and
mortar of the inner dome . The cross ties of the bottom chain can be seen protruding from the drum at the base of the dome . The others are hidden . Each stone
chain was supposed to be reinforced with a standard iron chain made of interlocking links , but a magnetic survey conducted in the 1970s failed to detect any
evidence of iron chains , which if they exist are deeply embedded in the thick masonry walls . Brunelleschi also included vertical `` ribs '' set on the corners
of the octagon , curving towards the center point . The Ribs , 13 feet ( 4 meters ) deep , are supported by 16 concealed ribs radiating from center . The ribs
had slits to take beams that supported platforms , thus allowing the work to progress upward without the need for scaffolding . A circular masonry dome can be
built without supports , called centering , because each course of bricks is a horizontal arch that resists compression .
</item 3>

<item 4>
Bed
A mortise or slot is cut vertically in the bedpost . Pins are inserted horizontally in the bed post so that the pins perpendicularly intersect the mortise . For
example , if one looked in the mortise , one might see part of one horizontal pin at the bottom of the mortise and a part of a second pin toward the top of the
mortise . Hooks are installed at the end of the rail . Usually these hooks are part of a plate that is attached to the rail . The hooks then are inserted into
the bed post mortise and hook over the pins . Plate - and - hook fastener . Instead of pins inserted horizontally into the bedpost , an eye plate ( post plate )
is installed on the bedpost . The hooks are installed on the rail , either as surface mount or recessed . Depending on the hardware , the bedpost may require a
mortise in order to allow the hooks to fasten to the plate . This is also referred to as a keyhole fastener , especially if the connector is more of a `` plug
'' than a `` hook '' . Bed bolts ( `` through - bolts '' ) are a different means of a knock - down connection .
</item 4>

<item 5>
Post and lintel
A. Burnham ( 2007 ) Jump up ^ L. Sprague De Camp , Ancient Engineers : Technology & Invention from the Earliest Times to the Renaissance ( U.S.A. : Barnes and
Noble , 1993 edition ) , 35 . References ( edit ) Summerson , John , The Classical Language of Architecture , 1980 edition , Thames and Hudson World of Art
series , ISBN 0500201773 Retrieved from `` https://en.wikipedia.org/w/index.php?title=Post_and_lintel&oldid=787548858 '' Categories : Architectural elements
Ancient Roman architectural elements Building Construction terminology Doors Windows Timber framing Structural system Talk Contents About Wikipedia فارسی
Gaeilge Edit links This page was last edited on 26 June 2017 , at 00 : 54 . Text is available under the Creative Commons Attribution - ShareAlike License ;
additional terms may apply . By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia
Foundation , Inc. , a non-profit organization . About Wikipedia
</item 5>

<item 6>
Post and lintel
See also ( edit ) Wikimedia Commons has media related to Post and Beam and Lintels . Architrave - structural lintel or beam resting on columns - pillars
Atalburu - Basque decorative lintel Dolmen - Neolithic megalithic tombs with structural stone lintels Dougong - traditional Chinese structural element I - beam
- steel lintels and beams Marriage stone - decorative lintel Timber framing - post and beam systems Structural design Opus caementicium Notes ( edit ) Jump up ^
Post and lintel is the main structural system in Northern China , the southern traditional timber buildings which use a column - and - tie structural system .
`` Structural Mechanism Of Southern Chinese Traditional Timber Frame Buildings '' SCIENCE CHINA Technological Sciences. 2011 , Vol 54 ( 7 )
http://www.redorbit.com/news/science/1112427956/structural-mechanism-of-southern-chinese-traditional-timber-frame-buildings/ Jump up ^ Summerson , 13 - 14 Jump
up ^ Summerson , 19 - 21 Jump up ^ C. Michael Hogan , Volubilis , Megalithic Portal , ed .
</item 6>

<item 7>
Stone Age
Kiva Standing stones megalith row Stonehenge Pyramid Dwellings Neolithic architecture British megalith architecture Nordic megalith architecture Burdei Cave
Cliff dwelling Dugout Hut Quiggly hole Jacal Longhouse Mud brick Mehrgarh Neolithic long house Pit - house Pueblitos Pueblo Rock shelter Blombos Cave Abri de la
Madeleine Sibudu Cave Stone roof Roundhouse Stilt house Alp pile dwellings Wattle and daub Water management Check dam Cistern Flush toilet Reservoir Water well
Other architecture Archaeological features Broch Burnt mound fulacht fiadh Causewayed enclosure Tor enclosure Circular enclosure Goseck Cursus Henge
Thornborough Oldest buildings Megalithic architectural elements Midden Timber circle Timber trackway Sweet Track Arts and culture Material goods Baskets
Beadwork Beds Chalcolithic Clothing / textiles timeline Cosmetics Glue Hides shoes Ötzi Jewelry amber use Mirrors Pottery Cardium Grooved ware Linear Jōmon
Unstan ware Sewing needle Weaving Wine Winery wine press Prehist Art Art of the Upper Paleolithic Art of the Middle Paleolithic Blombos Cave List of Stone Age
art Bird stone Bradshaw rock paintings Cairn
</item 7>

<item 8>
Post and lintel
As with the Roman temple portico front and its descendents in later classical architecture , trabeated features were often retained in parts of buildings as an
aesthetic choice . The classical orders of Greek origin were in particular retained in buildings designed to impress , even though they usually had little or no
structural role . Contents ( hide ) 1 Lintel beams 2 Engineering 3 See also 4 Notes 5 References Lintel beams ( edit ) Further information : Beam ( structure )
In architecture , a post-and - lintel or trabeated system refers to the use of horizontal beams or lintels which are borne up by columns or posts . The name is
from the Latin trabs , beam ; influenced by trabeatus , clothed in the trabea , a ritual garment . A noteworthy example of a trabeated system is in Volubilis ,
from the Roman era , where one side of the Decumanus Maximus is lined with trabeated elements , while the opposite side of the roadway is designed in arched
style . In India the style was used originally for wooden construction , but later the technique was adopted for stone structures for decorated load - bearing
and purely ornamented non-structural purposes .
</item 8>

<item 9>
Romanesque art
The Gloucester candlestick , early 12th century The bronze Gloucester candlestick and the brass font of 1108 -- 17 now in Liège are superb examples , very
different in style , of metal casting . The former is highly intricate and energetic , drawing on manuscript painting , while the font shows the Mosan style at
its most classical and majestic . The bronze doors , a triumphal column and other fittings at Hildesheim Cathedral , the Gniezno Doors , and the doors of the
Basilica di San Zeno in Verona are other substantial survivals . The aquamanile , a container for water to wash with , appears to have been introduced to Europe
in the 11th century . Artisans often gave the pieces fantastic zoomorphic forms ; surviving examples are mostly in brass . Many wax impressions from impressive
seals survive on charters and documents , although Romanesque coins are generally not of great aesthetic interest . The Cloisters Cross is an unusually large
ivory crucifix , with complex carving including many figures of prophets and others , which has been attributed to one of the relatively few artists whose name
is known , Master Hugo , who also illuminated manuscripts .
</item 9>

<item 10>
Post and lintel
Post and lintel - wikipedia Post and lintel Jump to : navigation , search Post and lintel construction of the World Heritage Monument site Airavatesvara Temple
, India Leinster House in Dublin retains columns under a pediment for aesthetic reasons ; the similar façade at the White House at least holds up a porch Timber
frame of a modern building under construction , also using trusses Stonehenge , an example of Neolithic architecture post and lintel construction . In
architecture , post and lintel ( also called prop and lintel or a trabeated system ) is a building system where strong horizontal elements are held up by strong
vertical elements with large spaces between them . This is usually used to hold up a roof , creating a largely open space beneath , for whatever use the
building is designed for . The horizontal elements are called by a variety of names including lintel , header , architrave or beam , and the supporting vertical
elements may be called columns , pillars , or posts . The use of wider elements at the top of the post , called capitals , to help spread the load , is common
to many traditions .
</item 10>
=== END ITEM id=85adde773634f6fa ===

=== ITEM id=630896f39dd7d536 ===
Gold sentences and facts:
Sentence 0: Deadpool 2 is scheduled to be released in the United States on May 18 , 2018 . A sequel , Deadpool 3 , is in development .
  Fact 0.0: Deadpool 2 is scheduled to be released in the United States on May 18, 2018.
  Fact 0.1: A sequel, Deadpool 3, is in development.

Retrieved items:
<item 1>
*Deadpool 2* was confirmed in 2016 with Ryan Reynolds and director Tim Miller returning, but reported creative disagreements led David Leitch to take over as
director. The film’s release was moved earlier, while *New Mutants* was delayed to 2019. *Deadpool 2* later underwent Vancouver reshoots under the working title
“Daisy”; reports said it scored higher than the original in test screenings and described material added during reshoots.
</item 1>

<item 2>
Deadpool 2’s release moved from June 1 to May 18, 2018. Six days of Vancouver reshoots under the working title Daisy added audience-favorite material,
especially featuring Cable and Domino; test scores reportedly reached 98/100, versus 91 for the original. Two hours of filming in Los Angeles captured a secret
cameo. Trailers revealed Terry Crews as Bedlam, Rob Delaney as Peter, Bill Skarsgård, and Lewis Tan as Sh
</item 2>

<item 3>
Deadpool 2
By April 26 , opening weekend industry projections had reached upwards of $150 million . Sequel By November 2016 , with development underway on Deadpool 2 , Fox
was also planning Deadpool 3 , which was said to include the team X-Force . With the confirmation that Leitch would direct Deadpool 2 , it was revealed that Fox
was looking for a separate filmmaker to direct Deadpool 3 . In March 2017 , Reese clarified that though Deadpool 2 sets - up the X-Force team , a future film
focused on the team would be separate from Deadpool 3 , `` so I think we 'll be able to take two paths . ( X-Force ) is where we 're launching something bigger
, but then ( Deadpool 3 is ) where we 're contracting and staying personal and small . '' After the proposed acquisition of 21st Century Fox by Disney was
announced in December 2017 , Disney CEO Bob Iger said that Deadpool would be integrated with the Marvel Cinematic Universe under Disney , and that the company
would be willing to make future R - rated Deadpool films `` as long as we let the audiences know what 's coming '' .
</item 3>

<item 4>
*Deadpool 2* was an upcoming 2018 American sequel and X-Men spin-off, categorized as a superhero action comedy, black comedy, and vigilante film. It was filmed
in Vancouver, set partly in Westchester County, and associated with IMAX, metafiction, and themes of cancer.
</item 4>

<item 5>
*Deadpool 2* follows Wade Wilson/Deadpool, played by Ryan Reynolds, as he forms the X-Force to protect young mutant Russell from time-traveling soldier Cable,
played by Josh Brolin. During filming, stunt performer Joi “SJ” Harris died in a motorcycle accident. The film was scheduled for U.S. release on May 18, 2018,
and *Deadpool 3* was in development.
</item 5>

<item 6>
*Deadpool 2* was developed after Fox green-lit a sequel with Ryan Reynolds and writers Rhett Reese and Paul Wernick. Tim Miller left amid creative differences,
and David Leitch directed. Filmed in British Columbia in 2017, it stars Reynolds, Josh Brolin as Cable, Zazie Beetz as Domino, and Julian Dennison as Russell;
Deadpool forms X-Force to protect Russell from Cable. Returning characters include Colossus, Negasonic Teenage
</item 6>

<item 7>
Deadpool 2
As part of a promotion for the film with beverage company Mike 's Harder , Sister Margaret 's School for Wayward Girls -- the fictional bar featured in the
films -- will be emulated in pop - ups at the Alligator Lounge in New York from April 26 -- 28 and the Slipper Clutch in Los Angeles from May 10 -- 12 . The pop
- ups will respectively serve pizza and chimichangas , traditionally Deadpool 's favorite food , as well as Deadpool - inspired Harder drinks . The campaign
also includes the chance to win a trip to the film 's premiere through Harder , and the Los Angeles pop - up will benefit the nonprofit DTLA Film Festival . Box
office projection Initial box office projections for the film in March 2018 expected it to gross $100 million over its opening weekend in the United States and
Canada , with a final domestic total expected to be lower than that of the first film ( $363 million ) at around $242 million . On April 20 , 2018 , both
Fandango and Regal Cinemas announced that Deadpool 2 was the best pre-selling R - rated film in their respective histories .
</item 7>

<item 8>
*Deadpool 2* is a 20th Century Fox superhero film directed by David Leitch, written by Rhett Reese, Paul Wernick, and Ryan Reynolds, and based on the Marvel
character created by Rob Liefeld and Fabian Nicieza. A sequel to *Deadpool* and the intended 11th X-Men film, it stars Reynolds, Josh Brolin, Morena Baccarin,
Julian Dennison, and Zazie Beetz. Its U
</item 8>

<item 9>
Deadpool 2
Deadpool 2 - wikipedia Deadpool 2 Jump to : navigation , search Deadpool 2 Teaser poster Directed by David Leitch Produced by Simon Kinberg Ryan Reynolds Lauren
Shuler Donner Written by Rhett Reese Paul Wernick Ryan Reynolds Based on Deadpool by Rob Liefeld Fabian Nicieza Starring Ryan Reynolds Josh Brolin Morena
Baccarin Julian Dennison Zazie Beetz T.J. Miller Brianna Hildebrand Jack Kesy Stefan Kapičić Music by Tyler Bates Cinematography Jonathan Sela Edited by Craig
Alpert Elísabet Ronaldsdóttir Production company Marvel Entertainment The Donners ' Company Genre Films Distributed by 20th Century Fox Release date May 18 ,
2018 ( 2018 - 05 - 18 ) ( United States ) Country United States Language English Deadpool 2 is an upcoming American superhero film based on the Marvel Comics
character Deadpool , distributed by 20th Century Fox . It is intended to be the eleventh installment in the X-Men film series , and a sequel to the 2016 film
Deadpool .
</item 9>

<item 10>
Deadpool 2
During filming , stunt woman Joi `` SJ '' Harris died in a motorcycle accident . Deadpool 2 is scheduled to be released in the United States on May 18 , 2018 .
A sequel , Deadpool 3 , is in development . Contents ( hide ) 1 Premise 2 Cast 3 Production 3.1 Development 3.2 Pre-production 3.3 Filming 3.4 Post-production 4
Music 5 Release 5.1 Marketing 6 Box office projection 7 Sequel 8 References 9 External links Premise Deadpool forms a team of mutants called the X-Force to
protect young mutant Russell from the time - traveling soldier Cable . Cast Ryan Reynolds as Wade Wilson / Deadpool : A wisecracking mercenary with accelerated
healing but severe scarring over his body after undergoing an experimental regenerative mutation . He forms the X-Force , a team of mutants . Josh Brolin as
Nathan Summers / Cable : A time travelling cybernetic mutant soldier , `` in many ways the opposite of Deadpool '' .
</item 10>
=== END ITEM id=630896f39dd7d536 ===

=== ITEM id=7fbe5c9af51088a5 ===
Gold sentences and facts:
Sentence 0: The 2017 -- 18 NBA season is the 72nd season of the National Basketball Association ( NBA ) . The regular season began on October 17 , 2017 , earlier than previous seasons to reduce the number of `` back - to - back '' games teams are scheduled to play , with the 2017 Eastern Conference champion ( and Finals runner -- up ) Cleveland Cavaliers hosting a game against the Boston Celtics at Quicken Loans Arena in Cleveland , Ohio Christmas games were played on December 25 , 2017 . The 2018 NBA All - Star Game was played on February 18 , 2018 , at the Staples Center in Los Angeles , California . LeBron James of the Cleveland Cavaliers was named the All - Star Game Most Valuable Player . The regular season ended on April 11 , 2018 and the playoffs began on April 14 , 2018 .
  Fact 0.0: The 2017-18 NBA season is the 72nd season of the National Basketball Association (NBA).
  Fact 0.1: The 2017-18 NBA regular season began on October 17, 2017.
  Fact 0.2: The 2017-18 NBA regular season began earlier than previous seasons.
  Fact 0.3: The earlier start of the 2017-18 season was meant to reduce the number of 'back-to-back' games teams are scheduled to play.
  Fact 0.4: The 2017 Eastern Conference champion Cleveland Cavaliers, who were also Finals runner-up, hosted a game against the Boston Celtics at Quicken Loans Arena in Cleveland, Ohio.
  Fact 0.5: Christmas games were played on December 25, 2017.
  Fact 0.6: The 2018 NBA All-Star Game was played on February 18, 2018.
  Fact 0.7: The 2018 NBA All-Star Game was played at the Staples Center in Los Angeles, California.
  Fact 0.8: LeBron James of the Cleveland Cavaliers was named the All-Star Game Most Valuable Player.
  Fact 0.9: The 2017-18 NBA regular season ended on April 11, 2018.
  Fact 0.10: The 2017-18 NBA playoffs began on April 14, 2018.
Sentence 1: 2017 -- 18 NBA season League National Basketball Association Sport Basketball Duration October 17 , 2017 -- April 11 , 2018 April 14 -- May 28 , 2018 ( Playoffs ) May 31 -- June 17 , 2018 ( Finals ) Number of games 82 Number of teams 30 TV partner ( s ) ABC , TNT , ESPN , NBA TV Draft Top draft pick Markelle Fultz Picked by Philadelphia 76ers Regular season Top seed Houston Rockets Top scorer James Harden ( Houston ) Playoffs Finals NBA seasons ← 2016 -- 17 2018 -- 19 →
  Fact 1.0: The league of the 2017-18 NBA season is the National Basketball Association.
  Fact 1.1: The sport of the 2017-18 NBA season is basketball.
  Fact 1.2: The 2017-18 NBA regular season ran from October 17, 2017 to April 11, 2018.
  Fact 1.3: The 2017-18 NBA playoffs ran from April 14 to May 28, 2018.
  Fact 1.4: The 2017-18 NBA Finals ran from May 31 to June 17, 2018.
  Fact 1.5: The 2017-18 NBA season had 82 games.
  Fact 1.6: The 2017-18 NBA season had 30 teams.
  Fact 1.7: ABC, TNT, ESPN and NBA TV were the TV partners of the 2017-18 NBA season.
  Fact 1.8: Markelle Fultz was the top draft pick, picked by the Philadelphia 76ers.
  Fact 1.9: The Houston Rockets were the top seed in the 2017-18 NBA regular season.
  Fact 1.10: James Harden of the Houston Rockets was the top scorer of the 2017-18 NBA season.
  Fact 1.11: The 2016-17 season is the season before the 2017-18 NBA season.
  Fact 1.12: The 2018-19 season is the season after the 2017-18 NBA season.

Retrieved items:
<item 1>
2017–18 NBA season
573 18.0 82 9 Denver Nuggets 46 36 . 561 19.0 82 10 Los Angeles Clippers 42 40 . 512 23.0 82 11 Los Angeles Lakers 35 47 . 427 30.0 82 12 Sacramento Kings 27 55
. 329 38.0 82 13 Dallas Mavericks 24 58 . 293 41.0 82 14 Memphis Grizzlies 22 60 . 268 43.0 82 15 Phoenix Suns 21 61 . 256 44.0 82 Notes z -- Clinched home
court advantage for the entire playoffs c -- Clinched home court advantage for the conference playoffs y -- Clinched division title x -- Clinched playoff spot *
-- Division leader Tiebreakers ( edit ) Eastern Conference Miami clinched # 6 seed over Milwaukee based on head - to - head sweep ( 3 -- 0 ) .
</item 1>

<item 2>
2017–18 NBA season
On December 16 , 2017 , Miami Heat head coach Erik Spoelstra surpassed Pat Riley for most wins in franchise history with his 455th career win . On December 18 ,
2017 , during halftime against the Golden State Warriors , the Los Angeles Lakers held a ceremony to retire both of Kobe Bryant 's jersey numbers in 8 and 24 .
That same day , the Naismith Basketball Hall of Fame updates the length for players , coaches , and referees eligible for entry into the Hall of Fame to
decrease by three years , allowing names like Steve Nash , Chauncey Billups , Ray Allen , and Jermaine O'Neal to potentially enter earlier than originally
expected . On December 25 , 2017 , in a win against the Cleveland Cavaliers , Draymond Green became the sixth player in NBA history to record a triple double on
Christmas Day . On January 1 , 2018 , DeMar DeRozan recorded the first 50 - point performance of the year against the Milwaukee Bucks . He also broke the
Toronto Raptors ' franchise record for points with 52 .
</item 2>

<item 3>
2017–18 NBA season
The regular season began on October 17 , 2017 , earlier than previous seasons to reduce the number of `` back - to - back '' games teams are scheduled to play ,
with the 2017 Eastern Conference champion ( and Finals runner -- up ) Cleveland Cavaliers hosting a game against the Boston Celtics at Quicken Loans Arena in
Cleveland , Ohio Christmas games were played on December 25 , 2017 . The 2018 NBA All - Star Game was played on February 18 , 2018 , at the Staples Center in
Los Angeles , California . LeBron James of the Cleveland Cavaliers was named the All - Star Game Most Valuable Player . The regular season ended on April 11 ,
2018 and the playoffs began on April 14 , 2018 . Contents ( hide ) 1 Transactions 1.1 Retirement 1.2 In - season trades 1.3 Free agency 1.4 Coaching changes
1.4. 1 Off - season 1.4. 2 In - season 2 Preseason 2.1 International games 3 Regular season 3.1 By conference 3.1.
</item 3>

<item 4>
2017–18 NBA season
92 1992 -- 93 1993 -- 94 1994 -- 95 1995 -- 96 1996 -- 97 1997 -- 98 1998 -- 99 2000s 1999 -- 00 2000 -- 01 2001 -- 02 2002 -- 03 2003 -- 04 2004 -- 05 2005 --
06 2006 -- 07 2007 -- 08 2008 -- 09 2010s 2009 -- 10 2010 -- 11 2011 -- 12 2012 -- 13 2013 -- 14 2014 -- 15 2015 -- 16 2016 -- 17 2017 -- 18 2018 -- 19 2017 --
18 NBA season by team Draft Summer League All - Star Game Playoffs Finals Transactions G League Eastern Atlantic Boston Brooklyn New York Philadelphia Toronto
Central Chicago Cleveland Detroit Indiana Milwaukee Southeast Atlanta Charlotte Miami Orlando Washington Western Northwest Denver Minnesota Oklahoma City
Portland Utah Pacific Golden State L.A. Clippers L.A.
</item 4>

<item 5>
2017–18 NBA season
As of March 6 , 2018 , there are 21 NBA teams that had signed uniform sponsorship contracts : Atlanta Hawks : Sharecare Boston Celtics : General Electric
Brooklyn Nets : Infor Charlotte Hornets : LendingTree Cleveland Cavaliers : Goodyear Tire and Rubber Company Dallas Mavericks : 5Miles Denver Nuggets : Western
Union Detroit Pistons : Flagstar Bank Golden State Warriors : Rakuten Los Angeles Clippers : Bumble Los Angeles Lakers : Wish Miami Heat : Ultimate Software
Milwaukee Bucks : Harley - Davidson Minnesota Timberwolves : Fitbit New Orleans Pelicans : Zatarain 's New York Knicks : Squarespace Orlando Magic : The Walt
Disney Company Philadelphia 76ers : StubHub Sacramento Kings : Blue Diamond Growers Toronto Raptors : Sun Life Financial Utah Jazz : Qualtrics Nine who do not :
Chicago Bulls , Houston Rockets , Indiana Pacers , Memphis Grizzlies , Oklahoma City Thunder , Phoenix Suns , Portland Trail Blazers , San Antonio Spurs and
Washington Wizards . Notable occurrences ( edit ) On July 6 , 2017 , the NBA unveiled an update to its logo , featuring new typography and a modified color
scheme . The regular season started eight days earlier than the previous season .
</item 5>

<item 6>
2017–18 NBA season
2017 -- 18 NBA season - wikipedia 2017 -- 18 NBA season Jump to : navigation , search 2017 -- 18 NBA season League National Basketball Association Sport
Basketball Duration October 17 , 2017 -- April 11 , 2018 April 14 -- May 28 , 2018 ( Playoffs ) May 31 -- June 17 , 2018 ( Finals ) Number of games 82 Number of
teams 30 TV partner ( s ) ABC , TNT , ESPN , NBA TV Draft Top draft pick Markelle Fultz Picked by Philadelphia 76ers Regular season Top seed Houston Rockets Top
scorer James Harden ( Houston ) Playoffs Finals NBA seasons ← 2016 -- 17 2018 -- 19 → The 2017 -- 18 NBA season is the 72nd season of the National Basketball
Association ( NBA ) .
</item 6>

<item 7>
2017–18 NBA season
The regular season started eight days earlier than the previous season . This move by the NBA aims for fewer `` back - to - back '' games , even fewer than the
previous season which had already reached an all - time low to help address teams that are `` resting '' their superstars . The NBA decreased the maximum number
of timeouts in a regulation game from 18 to 14 . The three team timeouts during the last two minutes of the game were also decreased to two per team in the last
three minutes to improve the flow and pace of the game . The trade deadline was moved from four days after the NBA All - Star Game to ten days before it , so
teams can settle their rosters before the All - Star break . Free - throw shooters will be assessed a delay - of - game violation if they venture outside of the
three - point line in between attempts . The NBA also added two - way contracted players , signifying an increasing partnership between the NBA and the newly
branded NBA G League . Referees can now assess dangerous closeouts of defenders to jump shooters with little space to land for a technical or a flagrant foul .
</item 7>

<item 8>
2017–18 NBA season
Africa 2017 Playoffs List Finals Champions History Predecessors BAA NBL ABA Merger Criticisms and controversies 2007 Tim Donaghy betting scandal Lockouts
Seasons Records regular season post-season All - Star Game Win - loss records Personalities Players Current rosters Foreign players Race and ethnicity First
overall draft picks Highest paid Retired numbers Banned or suspended NBPA Head coaches Current Player - coaches Champions Foreign coaches NBCA Owners Referees
Awards and honors Larry O'Brien Trophy NBA Awards NBA MVP Finals MVP All - Star Game MVP Hall of Fame Members NBA Silver Anniversary Team NBA 35th Anniversary
Team 50 Greatest Players Others Arenas Business Collective bargaining agreement Salary cap NBA Store Culture Cheerleading Mascots Dress code G League Midwest
Division Media TV NBA TV Rivalries Teams Defunct Expansion Relocated Timeline WNBA Basketball in the United States Category Portal 2017 -- 18 season Retrieved
from `` https://en.wikipedia.org/w/index.
</item 8>

<item 9>
2017–18 NBA season
Denver Minnesota Oklahoma City Portland Utah Pacific Golden State L.A. Clippers L.A. Lakers Phoenix Sacramento Southwest Dallas Houston Memphis New Orleans San
Antonio National Basketball Association Eastern Conference Atlantic Boston Celtics Brooklyn Nets New York Knicks Philadelphia 76ers Toronto Raptors Central
Chicago Bulls Cleveland Cavaliers Detroit Pistons Indiana Pacers Milwaukee Bucks Southeast Atlanta Hawks Charlotte Hornets Miami Heat Orlando Magic Washington
Wizards Western Conference Northwest Denver Nuggets Minnesota Timberwolves Oklahoma City Thunder Portland Trail Blazers Utah Jazz Pacific Golden State Warriors
Los Angeles Clippers Los Angeles Lakers Phoenix Suns Sacramento Kings Southwest Dallas Mavericks Houston Rockets Memphis Grizzlies New Orleans Pelicans San
Antonio Spurs Annual events Draft Eligibility Summer League Christmas Day All - Star Weekend Game Global Games Africa 2015 ,
</item 9>

<item 10>
Game seven
NBA.com . Retrieved 2012 - 05 - 25 . Jump up ^ `` Minneapolis Makes One Last Title Run '' . NBA.com . Retrieved 2012 - 05 - 25 . Jump up ^ `` Nats Win First
Title of Shot - Clock Era '' . NBA.com . Retrieved 2012 - 05 - 25 . Jump up ^ `` Celtics Officially Launch Their Dynasty '' . NBA.com . Retrieved 2012 - 05 - 25
. Jump up ^ `` Hawks Force Game 7 , But Celtics Repeat '' . NBA.com . Retrieved 2012 - 05 - 25 . Jump up ^ `` Celtics , Lakers Work OT to Start Rivalry '' .
NBA.com . Retrieved 2012 - 05 - 25 . Jump up ^ `` Red Lights Cigar to Close Coaching Career '' . NBA.com . Retrieved 2012 - 05 - 25 . Jump up ^ `` West 's
Heroics Ca n't Deprive Russell of 11th Title '' . NBA.com .
</item 10>
=== END ITEM id=7fbe5c9af51088a5 ===

=== ITEM id=ae02dd70fc07314c ===
Gold sentences and facts:
Sentence 0: `` Where Have All the Flowers Gone ? '' is a modern folk - style song . The melody and the first three verses were written by Pete Seeger in 1955 and published in Sing Out ! magazine . Additional verses were added in May 1960 by Joe Hickerson , who turned it into a circular song . Its rhetorical `` where ? '' and meditation on death place the song in the ubi sunt tradition . In 2010 , the New Statesman listed it as one of the `` Top 20 Political Songs '' .
  Fact 0.0: Where Have All the Flowers Gone? is a modern folk-style song.
  Fact 0.1: The melody of Where Have All the Flowers Gone? was written by Pete Seeger in 1955.
  Fact 0.2: The first three verses of Where Have All the Flowers Gone? were written by Pete Seeger in 1955.
  Fact 0.3: The melody of Where Have All the Flowers Gone? was published in Sing Out! magazine.
  Fact 0.4: The first three verses of Where Have All the Flowers Gone? were published in Sing Out! magazine.
  Fact 0.5: Additional verses of Where Have All the Flowers Gone? were added in May 1960 by Joe Hickerson.
  Fact 0.6: Joe Hickerson turned Where Have All the Flowers Gone? into a circular song.
  Fact 0.7: The rhetorical 'where?' and meditation on death of Where Have All the Flowers Gone? place the song in the ubi sunt tradition.
  Fact 0.8: In 2010, the New Statesman listed Where Have All the Flowers Gone? as one of the 'Top 20 Political Songs'.

Retrieved items:
<item 1>
Where Have All the Flowers Gone?
Version Title Artist Basque Loreak non dira ? Lou Topet , Harkaitz Cano Belarusian Дзе кветкі ўсе ? Bar Akaryna Bengali কোথায় গেল তারা ? ( Kothay Gelo Tara )
Kabir Suman Catalan Què se n'ha fet d'aquelles flors ? Roslyn Smith , La Marta ( Club Super3 ) Chinese 花 兒 怎麼 不見 了 ? Poon Sow Keng ( 潘秀瓊 ) Croatian Kamo je
cvijeće otišlo ? Monia Verardi Czech Řekni , kde ty kytky jsou Judita Čeřovská , Marie Rottrová , Marta Kubišová Danish Where Have All the Flowers Gone Savage
Rose Dutch Zeg me waar de bloemen zijn Jaap Fischer English Where have all the flowers gone ?
</item 1>

<item 2>
Where Have All the Flowers Gone?
Seeger created a song which was subsequently published in Sing Out in 1962 . He recorded a version with three verses on The Rainbow Quest album ( Folkways LP FA
2454 ) released in July 1960 . Later , Joe Hickerson added two more verses with a recapitulation of the first in May 1960 in Bloomington , Indiana . In 2010 ,
the New Statesman listed it as one of the `` Top 20 Political Songs '' . The song appeared on the compilation album Pete Seeger 's Greatest Hits ( 1967 )
released by Columbia Records as CS 9416 . Pete Seeger 's recording from the Columbia album The Bitter and the Sweet ( November 1962 ) , CL 1916 , produced by
John H. Hammond was also released as a Columbia Hall of Fame 45 single as 13 - 33088 backed by `` Little Boxes '' in August , 1965 . Versions ( edit ) The
Kingston Trio recorded the song in 1961 .
</item 2>

<item 3>
Where Have All the Flowers Gone?
Sing Out ! . 53 ( 2 ) : 76 . ^ Jump up to : Joe Hickerson . `` Where Have All the Flowers Gone ? '' . presentation to SEM ( Society for Ethnomusicology ) , 50th
Annual Meeting in Atlanta ( quoted in thread ) . Mudcat.org . Retrieved October 20 , 2009 . ^ Jump up to : Smith , Ian K ( March 25 , 2010 ) . `` Top 20
Political Songs : Where Have All the Flowers Gone '' . New Statesman . Retrieved March 25 , 2010 . Jump up ^ A Folk Legend 's Fertile Ground , Oberlin Alumni
Magazine , summer 2014 Jump up ^ Notes from : Where Have All the Flowers Gone - The Songs of Pete Seeger ^ Jump up to : Dunaway , David King ( 2008 ) . How Can
I Keep From Singing ? The Ballad of Pete Seeger , pp. 228 - 30 . Random House , Inc . ISBN 0 - 345 - 50608 - 1 .
</item 3>

<item 4>
Where Have All the Flowers Gone?
Where Have All the Flowers Gone ? - Wikipedia Where Have All the Flowers Gone ? Jump to : navigation , search For the 2008 documentary , see Where Have All the
Flowers Gone ? ( film ) . 1960 release on The Rainbow Quest LP , Folkways , FA 2454 , by Pete Seeger Columbia Hall of Fame 45 rpm single release as 13 - 33088
featuring the November 1962 version . August 1965 . `` Where Have All The Flowers Gone ? '' Song Language English Composer ( s ) Pete Seeger Lyricist ( s ) Pete
Seeger and Joe Hickerson `` Where Have All the Flowers Gone ? '' is a modern folk - style song . The melody and the first three verses were written by Pete
Seeger in 1955 and published in Sing Out ! magazine . Additional verses were added in May 1960 by Joe Hickerson , who turned it into a circular song . Its
rhetorical `` where ? '' and meditation on death place the song in the ubi sunt tradition . In 2010 , the New Statesman listed it as one of the `` Top 20
Political Songs '' .
</item 4>

<item 5>
Where Have All the Flowers Gone?
ISBN 0 - 345 - 50608 - 1 . Jump up ^ Pete Seeger - Little Boxes / Where Have All the Flowers Gone . Discogs.com . Jump up ^ Pete Seeger - The Bitter and the
Sweet . Discogs.com . Jump up ^ Where Have All the Flowers Gone ? Second Hand Songs . Jump up ^ The Kingston Trio time line . The single was released on
December 18 , 1961 by the group on Capitol Records as a 45 single , 4671 . Source : Liner notes for The Kingston Trio : The Capital Years ( Capitol Records
CD7243 8 28498 2 7 ) . Jump up ^ Gilliland , John ( 1969 ) . `` Show 18 - Blowin ' in the Wind : Pop discovers folk music . ( Part 1 ) '' ( audio ) . Pop
Chronicles . University of North Texas Libraries . Track 5 . Jump up ^ Whitburn , Joel ( 2002 ) . Top Adult Contemporary : 1961 - 2001 . Record Research . p.
137 .
</item 5>

<item 6>
Where Have All the Flowers Gone?
Oğuz Tarihmen Ukrainian Де всі квіти , розкажи De vsi kvity , rozkazhy Maria Burmaka Ukrainian Квіти де ? Kvity De ? Yana Zavarzina Grammy Hall of Fame ( edit )
Pete Seeger 's recording of his composition was inducted into the Grammy Hall of Fame , which is a special Grammy award established in 1973 to honor recordings
that are at least 25 years old and that have `` qualitative or historical significance . '' Pete Seeger : Grammy Hall of Fame Awards Year recorded Title Genre
Label Year inducted 1964 `` Where Have All the Flowers Gone ? '' Folk ( single ) Columbia 2002 See also ( edit ) List of anti-war songs References ( edit ) Jump
up ^ Seeger , Pete . `` Where Have All the Flowers Gone '' . Sing Out ! . 11 ( 5 ) . Jump up ^ Hickerson , Joe ( 2009 -- 2010 ) . `` The Songfinder '' . Sing
Out ! . 53 ( 2 ) : 76 .
</item 6>

<item 7>
Where Have All the Flowers Gone?
Record Research . p. 137 . Jump up ^ `` Marlene Dietrich , Actor / Singer '' . Answers.com . Retrieved 7 April 2010 . Jump up ^ `` Marlene Dietrich - Sag mir wo
die Blumen sind '' . Offiziellecharts.de . Retrieved 27 June 2016 . Jump up ^ Album title En kväll med Towa och Lasse , track 8 , record label Teldec , Hamburg
, cataloged in the Swedish Media Database ( SMDb ) of the Royal Library in Stockholm . Jump up ^ Inga blommor finns det mer , blog post by Enn Kokk , without
date , read January 20 , 2014 . Jump up ^ Grady Martin . Instrumentally Yours . AllMusic.com . Jump up ^ Grammy Hall of Fame Database . Bibliography ( edit )
Seeger , Pete ; Blood , Peter ( 1993 ) . Where have all the flowers gone : a singer 's stories , songs , seeds , robberies ( book ) format = requires url = (
help ) ( Biography ) . Bethlehem , Pennsylvania : Sing Out !
</item 7>

<item 8>
Where Have All the Flowers Gone?
Versions ( edit ) The Kingston Trio recorded the song in 1961 . Believing it to be a traditional song , they claimed authorship , although upon notice from
Seeger they had their name removed and credited Seeger and Hickerson . Seeger acknowledged their success with this song . Their single , with `` O Ken Karanga
'' as the A-side and the hit `` Where Have All The Flowers Gone ? '' the B - side , reached # 21 in the 1962 Billboard Hot 100 chart and # 4 on the Easy
Listening chart . Peter , Paul and Mary included the song on their eponymous debut album ( which spent five weeks as the # 1 album in the country ) in 1962 .
Marlene Dietrich performed the song in English , French , and German . The song was first performed in French ( as `` Qui peut dire où vont les fleurs ? '' ) by
Marlene in 1962 at a UNICEF concert . She also recorded the song in English and in German , the latter titled `` Sag ' mir , wo die Blumen sind '' , with lyrics
translated by Max Colpet .
</item 8>

<item 9>
Where Have All the Flowers Gone?
Bethlehem , Pennsylvania : Sing Out ! . pp. 166 -- 169 . ISBN 978 - 1 - 881322 - 01 - 6 . OCLC 28150656 . Pete Seeger Studio albums American Folk Songs for
Children God Bless the Grass Dangerous Songs ! ? Rainbow Race At 89 Pete Remembers Woody Live albums We Shall Overcome Songs `` The Bells of Rhymney '' `` God
's Counting on Me , God 's Counting on You '' `` Gotta Travel On '' `` If I Had a Hammer '' `` If You Miss Me at the Back of the Bus '' `` Little Boxes '' `` My
Rainbow Race '' `` Pittsburgh Town '' `` Turn ! Turn ! Turn ! '' `` Waist Deep in the Big Muddy `` We Shall Overcome '' `` Where Have All the Flowers Gone ? ''
Almanac Singers Songs for John Doe Talking Union Deep Sea Chanteys and Whaling Ballads Sod Buster Ballads Songs of the Lincoln Battalion Dear Mr. President The
Weavers The Weavers at Carnegie Hall The Weavers at Carnegie Hall Vol.
</item 9>

<item 10>
Where Have All the Flowers Gone?
The 1964 release of the song as a Columbia Records 45 single , 13 - 33088 , by Pete Seeger was inducted into the Grammy Hall of Fame in 2002 in the Folk
category . Contents ( hide ) 1 Composition 2 Versions 3 Grammy Hall of Fame 4 See also 5 References 6 Bibliography Composition ( edit ) Seeger found inspiration
for the song in October 1955 while he was on a plane bound for a concert at Oberlin College , one of the few venues which would hire him during the McCarthy era
. Leafing through his notebook he saw the passage , `` Where are the flowers , the girls have plucked them . Where are the girls , they 've all taken husbands .
Where are the men , they 're all in the army . '' These lines were taken from the traditional Cossack folk song `` Koloda - Duda '' , referenced in the Mikhail
Sholokhov novel And Quiet Flows the Don ( 1934 ) , which Seeger had read `` at least a year or two before '' . Seeger created a song which was subsequently
published in Sing Out in 1962 .
</item 10>
=== END ITEM id=ae02dd70fc07314c ===
