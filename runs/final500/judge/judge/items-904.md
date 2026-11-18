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

=== ITEM id=803e612a30ec5b50 ===
Gold sentences and facts:
Sentence 0: The original Princess Maru incarnation of the character made her cinematic debut in the 2017 film Wonder Woman , portrayed by Spanish actress Elena Anaya .
  Fact 0.0: The original Princess Maru incarnation of the character made her cinematic debut in the 2017 film Wonder Woman.
  Fact 0.1: The original Princess Maru incarnation of the character was portrayed by Spanish actress Elena Anaya in Wonder Woman.
  Fact 0.2: Elena Anaya is a Spanish actress.
Sentence 1: Doctor Poison appears in the DC Extended Universe film Wonder Woman ( 2017 ) , portrayed by Elena Anaya . In this adaptation , her identity is Isabel Maru , a Spanish chemist recruited by General Erich Ludendorff to create chemical weapons for the German army . Due to facial disfigurement , she wears a ceramic mask over the left side of her face . It marks the character 's first live action appearance , although the setting is World War I rather than the World War II setting of the comic books . When Steve Trevor steals a book of Doctor Maru 's notes on her new chemical weapons that would render the gas masks useless against them , he crash - lands on Themyscira during his escape , which alerts Diana to the current war . She initially believes that Ares is directly responsible for the war , but later learns that all he did was provide a few ideas for the weapons to the likes of Maru while the decision to use the weapons was still the choice of the humans themselves . Ares attempts to sway Diana to his point of view by offering her a chance to kill Maru for her sins and part in the war , but Diana rejects his offer upon realizing of Trevor 's sacrifice that inspires her to recognize that she must accept humanity for its flaws rather than condemn it for not being perfect . This leads Diana to spare Maru out of mercy , who then flees .
  Fact 1.0: Doctor Poison appears in the DC Extended Universe film Wonder Woman (2017).
  Fact 1.1: Doctor Poison is portrayed by Elena Anaya in Wonder Woman (2017).
  Fact 1.2: In Wonder Woman (2017), Doctor Poison's identity is Isabel Maru.
  Fact 1.3: In Wonder Woman (2017), Isabel Maru is a Spanish chemist.
  Fact 1.4: In Wonder Woman (2017), Isabel Maru was recruited by General Erich Ludendorff.
  Fact 1.5: General Erich Ludendorff recruited Isabel Maru to create chemical weapons for the German army.
  Fact 1.6: Due to facial disfigurement, Isabel Maru wears a ceramic mask over the left side of her face.
  Fact 1.7: The Wonder Woman (2017) portrayal marks the character's first live action appearance.
  Fact 1.8: In the Wonder Woman (2017) portrayal, the setting is World War I rather than the World War II setting of the comic books.
  Fact 1.9: When Steve Trevor steals a book of Doctor Maru's notes on her new chemical weapons, he crash-lands on Themyscira during his escape.
  Fact 1.10: Doctor Maru's notes describe new chemical weapons that would render the gas masks useless against them.
  Fact 1.11: Steve Trevor's crash-landing on Themyscira alerts Diana to the current war.
  Fact 1.12: Diana initially believes that Ares is directly responsible for the war.
  Fact 1.13: Diana later learns that Ares only provided a few ideas for the weapons to the likes of Maru.
  Fact 1.14: Diana later learns that the decision to use the weapons was still the choice of the humans themselves.
  Fact 1.15: Ares attempts to sway Diana to his point of view by offering her a chance to kill Maru for her sins and part in the war.
  Fact 1.16: Diana rejects Ares's offer upon realizing Steve Trevor's sacrifice.
  Fact 1.17: Steve Trevor's sacrifice inspires Diana to recognize that she must accept humanity for its flaws rather than condemn it for not being perfect.
  Fact 1.18: Diana spares Maru out of mercy.
  Fact 1.19: Maru flees after Diana spares her.

Retrieved items:
<item 1>
Doctor Poison
Doctor Poison - Wikipedia Doctor Poison Jump to : navigation , search This article needs to be updated . Please update this article to reflect recent events or
newly available information . ( June 2017 ) Doctor Poison Doctor Poison debuts in Sensation Comics # 2 . Publication information Publisher DC Comics First
appearance ( Princess Maru ) Sensation Comics # 2 ( February 1942 ) ( Marina Maru ) Wonder Woman # 151 ( Vol. 2 ) ( December 1999 ) Created by William Moulton
Marston Harry G. Peter In - story information Alter ego Princess Maru Marina Maru Team affiliations Villainy Inc . Secret Society of Super-Villains Godwatch
Abilities Scientific genius in fields of chemistry and biology Engineering of pathogens , poisons , toxins , plagues and chemical - based weapons . Doctor
Poison is the name of two fictional characters , supervillains who appear in DC Comics publications and related media . Both villains were members of the
super-villain team Villainy Inc. and have appeared as major recurring enemies for Wonder Woman .
</item 1>

<item 2>
Doctor Poison
Peter Other contributors Characters Wonder Women Diana Prince Orana Artemis of Bana - Mighdall Hippolyta Donna Troy Supporting characters Antiope Aphrodite
Artemis Artemis of Bana - Mighdall Batman Drusilla Etta Candy Fury Hephaestus Hera Heracles / Hercules Hermes I Ching Julia and Vanessa Kapatelis Mala Nemesis (
Thomas Tresser ) Nubia The Olympian Orion Paula Philippus Poseidon Queen Hippolyta Helena Sandsmark Sarge Steel Superman Steve Trevor Wonder Girl ( Cassie
Sandsmark Donna Troy ) Zeus Zola Adversaries Aegeus Angle Man Ares / Mars Baron Blitzkrieg Baroness Paula von Gunther Blue Snowman Veronica Cale Captain Wonder
Cheetah Circe Dark Angel Decay Doctor Cyber Doctor Poison Doctor Psycho Duke of Deception Earl of Greed Egg Fu / Chang Tzu Eviless The First Born Genocide
Giganta Hades Hypnota Kung Lord Conquest Mask Medusa Minister Blizzard Osira Queen Clea Red Panzer Silver Swan Tezcatlipoca Zara Factions Amazons of Themyscira
Amazons of Bana - Mighdall Children of Ares Godwatch Gorilla Knights Olympian Gods
</item 2>

<item 3>
Doctor Poison
and have appeared as major recurring enemies for Wonder Woman . The original Princess Maru incarnation of the character made her cinematic debut in the 2017
film Wonder Woman , portrayed by Spanish actress Elena Anaya . Contents ( hide ) 1 Publication history 2 Fictional character biography 2.1 Princess Maru 2.1. 1
Golden Age 2.1. 2 Post-Crisis 2.2 Marina Maru 2.2. 1 Post-Crisis 2.2. 2 The New 52 2.2. 3 DC Rebirth 3 Powers and abilities 4 Other versions 4.1 Wednesday
Comics 4.2 DC Super Friends 5 In other media 5.1 Television 5.2 Film 5.3 Video games 6 See also 7 References 8 External links Publication history ( edit ) The
Princess Maru incarnation of Doctor Poison first appeared in Sensation Comics # 2 and was created by William Moulton Marston and Harry G. Peter . The second
Doctor Poison first appeared in Wonder Woman Vol. 2 # 151 and was created by Eric Luke and Matthew Clark .
</item 3>

<item 4>
Doctor Poison
In other media ( edit ) Television ( edit ) A version of Doctor Poison appears in the Batman : The Brave and the Bold episode `` Joker : The Vile and the
Villainous . '' She is seen working as a bartender at a tavern populated by D - list supervillains when Joker pays a visit to it and meets Weeper . Film ( edit
) Doctor Poison appears in the DC Extended Universe film Wonder Woman ( 2017 ) , portrayed by Elena Anaya . In this adaptation , her identity is Isabel Maru , a
Spanish chemist recruited by General Erich Ludendorff to create chemical weapons for the German army . Due to facial disfigurement , she wears a ceramic mask
over the left side of her face . It marks the character 's first live action appearance , although the setting is World War I rather than the World War II
setting of the comic books . When Steve Trevor steals a book of Doctor Maru 's notes on her new chemical weapons that would render the gas masks useless against
them , he crash - lands on Themyscira during his escape , which alerts Diana to the current war .
</item 4>

<item 5>
Doctor Poison
5 , # 13 ( Dec 2016 ) Jump up ^ Wonder Woman vol. 5 , # 12 ( Dec 2016 ) Jump up ^ Wednesday Comics # 2 ( 2009 ) Jump up ^ DC Super Friends # 24 ( 2010 ) Jump up
^ O'Hara , Helen ( April 2017 ) . `` Alpha Female '' . Empire . pp. 60 -- 67 . access - date = requires url = ( help ) Jump up ^
https://community.wbgames.com/t5/Update-Notes/DC-Legends-Update-1-12/td-p/1519906 Beatty , Scott ( 2009 ) . Wonder Woman : The Ultimate Guide To The Amazon
Princess . Dorling Kindersley Publishing . pp. 82 -- 83 . ISBN 0 - 7894 - 9616 - X . External links ( edit ) Doctor Poison Rapsheet Gay League Profile The
Unofficial Doctor Poison Biography ( hide ) Wonder Woman William Moulton Marston Elizabeth Holloway Marston H.G. Peter Other contributors Characters Wonder
Women Diana Prince Orana Artemis of Bana - Mighdall
</item 5>

<item 6>
Doctor Poison
2 # 151 and was created by Eric Luke and Matthew Clark . Fictional character biography ( edit ) Princess Maru ( edit ) Golden Age ( edit ) Doctor Poison first
appeared as the chief of the poison division for a Nazi spy band who had planned to contaminate the United States Army 's water with `` Reverso , '' a drug that
compels whoever takes it to `` do the exact opposite of what they are told . '' She disguised her sex by wearing a bulky hooded costume and a mask . Doctor
Poison 's underlings captured Steve Trevor and brought him to their base in America where he was questioned by Doctor Poison . Wonder Woman , disguised as a
nurse , aided Steve Trevor but was forced to flee the scene after a lengthy battle . Meanwhile , the Reverso drug was successful in turning thousands of
American soldiers against their superiors . Wonder Woman recruited Etta Candy to help her create a diversion for Doctor Poison 's troops , which led to the
defeat of the villain . When Wonder Woman pulled her disguise off , she discovered Doctor Poison to be Maru , a Japanese princess .
</item 6>

<item 7>
Doctor Poison
DC Rebirth ( edit ) After the events of DC Rebirth , Doctor Poison 's history had been altered . In the current continuity , she is Colonel Marina Maru , a
Japanese soldier working for the organization called Poison which had been founded by her family . During Wonder Woman 's first few months in the United States
, the amazon discovered a group of men infected with the Maru Virus , a poison that drives its victims to rage - induced murder . Ten years later after Wonder
Woman had discovered she had been living a lie for several years , Veronica Cale sent Doctor Poison to attack the Amazon . Powers and abilities ( edit ) Doctor
Poison is an expert in the use of poisons , toxins , and plagues . Other versions ( edit ) Wednesday Comics ( edit ) Doctor Poison appears in the Wonder Woman
sections of anthology series Wednesday Comics . DC Super Friends ( edit ) Doctor Poison is among several scientist - themed super-villains who band together to
fight the Super Friends in the story `` Weird Science '' from the series DC Super Friends .
</item 7>

<item 8>
Doctor Poison
Teen Titans Justice League Action Wonder Woman ( 2017 film ) Justice League Miscellaneous Alternative versions Earth - Two Bizarra Olive Byrne Cultural impact
Professor Marston and the Wonder Women Literature Wonder Women ! The Untold Story of American Superheroines Portal Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Doctor_Poison&oldid=835962156 '' Categories : Wonder Woman characters Characters created by William Moulton Marston
Characters created by H.G. Peter Comics characters introduced in 1942 Comics characters introduced in 1999 DC Comics supervillains Golden Age supervillains
Female characters in comics Female characters in film Fictional princesses Fictional mass murderers Fictional colonels Fictional female doctors Fictional
Japanese military personnel Fictional chemists Fictional pathologists Fictional toxicologists Fictional virologists Nazis in comic book fiction Hidden
categories : Pages using citations with accessdate and no URL Wikipedia articles in need of updating from June 2017 All Wikipedia articles in need of updating
Character pop Converting comics character infoboxes Talk Contents About Wikipedia Español 한국어 Simple English Tagalog Edit links This page was last edited on 11
April 2018 , at 20 : 42 .
</item 8>

<item 9>
Doctor Poison
She initially believes that Ares is directly responsible for the war , but later learns that all he did was provide a few ideas for the weapons to the likes of
Maru while the decision to use the weapons was still the choice of the humans themselves . Ares attempts to sway Diana to his point of view by offering her a
chance to kill Maru for her sins and part in the war , but Diana rejects his offer upon realizing of Trevor 's sacrifice that inspires her to recognize that she
must accept humanity for its flaws rather than condemn it for not being perfect . This leads Diana to spare Maru out of mercy , who then flees . Video games (
edit ) Doctor Poison appears as a playable character in the mobile game DC Legends . See also ( edit ) List of Wonder Woman enemies References ( edit ) Jump up
^ Sensation Comics # 2 ( 1942 ) Jump up ^ Sensation Comics # 24 ( 1943 ) Jump up ^ Wonder Woman # 28 ( 1948 ) Jump up ^ Wonder Woman : Our Worlds at War # 1
Jump up ^ Wonder Woman # 81 Jump up ^ Wonder Woman # 48 ( Mar 2016 ) Jump up ^ Wonder Woman vol.
</item 9>

<item 10>
Doctor Poison
( 2006 -- 07 ) Amazons Attack ! ( 2007 ) The Circle ( 2008 ) Ends of the Earth ( 2008 ) Rise of the Olympian ( 2009 ) Flashpoint ( 2011 ) The Lies ( 2016 ) Year
One ( 2016 ) The Truth ( 2017 ) Godwatch ( 2017 ) Technology Bracelets Golden Girdle of Gaea Invisible plane Lasso of Truth Mental radio Pegasi Purple Ray Sky
Kangas In other media Super Friends ( episodes ) Wonder Woman ( 1974 film ) Wonder Woman ( TV series ) ( episodes ) Justice League ( episodes ) Justice League
Unlimited ( episodes ) Justice League : The New Frontier Wonder Woman ( 2009 film ) Justice League : Crisis on Two Earths Superman / Batman : Apocalypse Wonder
Woman ( 2011 TV pilot ) Young Justice DC Universe Online Justice League : Doom Justice League : The Flashpoint Paradox Justice League : War The Lego Movie
Justice League : Throne of Atlantis Justice League : Gods and Monsters Batman v Superman : Dawn of Justice Justice League vs.
</item 10>
=== END ITEM id=803e612a30ec5b50 ===

=== ITEM id=74c980afe5297912 ===
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
Michael Gregson is about to depart for Germany ; if he becomes a German citizen , he may be able to divorce his insane wife . Lady Rosamund discovers that Edith
spent the night with Gregson , and warns her about damaging her reputation . Lord Gillingham follows Mary back from London and proposes marriage . She refuses ,
saying she is still grieving Matthew 's death . 30 5 `` Episode Five '' Philip John Julian Fellowes 20 October 2013 ( 2013 - 10 - 20 ) 11.39 May 1922 . Alfred
applies to a chef training program at The Ritz , under Escoffier , but is turned down . Carson considers re-hiring Mr Molesley if Alfred leaves , but Molesley ,
unhappy at being demoted to a footman , dithers too long over the offer , and loses the opportunity . Thomas pressures Cora 's new lady 's maid , Baxter , to
spy on everyone for him . When a tenant farmer dies owing significant back rent , Mary and Tom want to end the lease and run the farm themselves .
</item 1>

<item 2>
List of Downton Abbey episodes
45 Christmas special No . overall Title Directed by Written by Original air date UK viewers ( millions ) 16 `` Christmas at Downton Abbey '' Brian Percival
Julian Fellowes 25 December 2011 ( 2011 - 12 - 25 ) 12.11 Series 3 ( 2012 ) ( edit ) ITV confirmed the commissioning of a third series on 3 November 2011 . It
entered production in early 2012 , and began airing 16 September 2012 on ITV and aired in the United States beginning 6 January 2013 . Series 3 No . overall No.
in series Title Directed by Written by Original air date UK viewers ( millions ) 17 `` Episode One '' Brian Percival Julian Fellowes 16 September 2012 ( 2012 -
09 - 16 ) 11.43 7.9 ( PBS ) March / April 1920 . Cora 's mother , Martha Levinson , sweeps into Downton for Matthew and Mary 's wedding , aggravating the
Dowager Countess ( among others ) with her critical and `` revolutionary '' American way of seeing things .
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
33 11 `` Episode Four '' Brian Kelly Julian Fellowes 9 October 2011 ( 2011 - 10 - 09 ) 11.30 12 5 `` Episode Five '' Brian Kelly Julian Fellowes 16 October 2011
( 2011 - 10 - 16 ) 11.59 13 6 `` Episode Six '' Andy Goddard Julian Fellowes 23 October 2011 ( 2011 - 10 - 23 ) 11.33 14 7 `` Episode Seven '' James Strong
Julian Fellowes 30 October 2011 ( 2011 - 10 - 30 ) 12.26 15 8 `` Episode Eight '' James Strong Julian Fellowes 6 November 2011 ( 2011 - 11 - 06 ) 12.45
Christmas special No .
</item 4>

<item 5>
List of Downton Abbey episodes
List of Downton Abbey episodes - Wikipedia List of Downton Abbey episodes Jump to : navigation , search This article 's plot summaries may be too long or
excessively detailed . Please help improve them by removing unnecessary details and making them more concise . ( January 2016 ) ( Learn how and when to remove
this template message ) Downton Abbey is a British period drama television series created by Julian Fellowes and co-produced by Carnival Films and Masterpiece .
It first aired on ITV in the United Kingdom on 26 September 2010 , and on PBS in the United States on 9 January 2011 , as part of the Masterpiece Classic
anthology . Six series have been made , the sixth airing in the autumn of 2015 in the UK and Ireland , and in January 2016 in the United States . On 26 March
2015 , the sixth series was confirmed to be the final series , with the final episode airing in the UK on 25 December 2015 on ITV . During the course of the
programme , 52 episodes of Downton Abbey aired over six series .
</item 5>

<item 6>
List of Downton Abbey episodes
68 8 ( + 1 ) 16 September 2012 ( 2012 - 09 - 16 ) 4 November 2012 ( 2012 - 11 - 04 ) 25 December 2012 ( special ) 11.91 8 ( + 1 ) 22 September 2013 ( 2013 - 09
- 22 ) 10 November 2013 ( 2013 - 11 - 10 ) 25 December 2013 ( special ) 11.84 5 8 ( + 1 ) 21 September 2014 ( 2014 - 09 - 21 ) 9 November 2014 ( 2014 - 11 - 09
) 25 December 2014 ( special ) 10.40 6 8 ( + 1 ) 20 September 2015 ( 2015 - 09 - 20 ) 8 November 2015 ( 2015 - 11 - 08 ) 25 December 2015 ( special ) 10.
</item 6>

<item 7>
List of Downton Abbey episodes
Christmas special No . overall Title Directed by Written by Original air date UK viewers ( millions ) 52 `` The Finale '' Michael Engler Julian Fellowes 25
December 2015 ( 2015 - 12 - 25 ) 11.22 29 September -- 31 December 1925 : Carson fumbles serving dinner when his hands begin shaking , suffering from an
unspecified hereditary disease his family called ' the palsy ' . Thomas secures a job with an elderly couple with a staff of only three , but is unhappy there .
Edith offers to expand Spratt 's magazine column to a whole page ; Denker discovers Spratt 's secret , and tries to get him dismissed by revealing it to the
Dowager Countess , but she finds the column jolly amusing and keeps him on . Molesley leaves service when the Headmaster offers him a teaching position , plus a
cottage ; upon parting , he and Baxter finally express their mutual romantic affection . Lord Merton tells Isobel he suffers from pernicious anaemia , but still
wants to marry her .
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
List of Downton Abbey episodes
With the future of Downton hanging in the balance , Mary and the Dowager Countess conspire to show Downton at its grandest to persuade Mrs Levinson to give
Downton more money and save them from financial ruin . Their attempts are impeded by the oven breaking down , giving Mrs Levinson a chance to take over running
the party herself in radical fashion . She reveals , when finally asked for help , that though she will `` entertain them in New York '' she can not provide
them with money , as her own late husband tied it up in the estate , much as Violet 's did . She tells them they have to adapt to the changing world , and
states her intent to return to America . Meanwhile downstairs , O'Brien and Thomas wage war against one another due to Alfred 's recent promotion to valet to
Matthew , whilst Mrs Levinson 's maid takes a shine to Alfred . Mrs Hughes faces a cancer scare that she shares only with Mrs Patmore and Doctor Clarkson , and
Isobel Crawley finds that Ethel has become a prostitute .
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
=== END ITEM id=74c980afe5297912 ===

=== ITEM id=664fd91dc7152e0c ===
Gold sentences and facts:
Sentence 0: No . Name Field Affiliation Date of Appointment Date of Retirement Roopa Ganguly Art Bharatiya Janata Party 04 - Oct - 2016 03 - Oct - 2022 Sambhaji Raje Social work Bharatiya Janata Party 07 - Jun - 2016 03 - May - 2022 Suresh Gopi Art Bharatiya Janata Party 25 - Apr - 2016 24 - Apr - 2022 Subramanian Swamy Economics Bharatiya Janata Party 25 - Apr - 2016 24 - Apr - 2022 5 Narendra Jadhav Economics Nominated 25 - Apr - 2016 24 - Apr - 2022 6 Mary Kom Sport Nominated 25 - Apr - 2016 24 - Apr - 2022 7 Swapan Dasgupta Journalism Nominated 25 - Apr - 2016 24 - Apr - 2022 8 K.T.S. Tulsi Law Nominated 25 - Feb - 2014 24 - Feb - 2020 9 K. Parasaran Law Nominated 09 - Jun - 2012 28 - Jun - 2018 10 Rekha Art Nominated 27 - Apr - 2012 26 - Apr - 2018 11 Sachin Tendulkar Social service Nominated 27 - Apr - 2012 26 - Apr - 2018 12 Anu Aga Business Nominated 27 - Apr - 2012 26 - Apr - 2018
  Fact 0.0: Roopa Ganguly's field is Art and affiliation is Bharatiya Janata Party, with appointment on 04-Oct-2016 and retirement on 03-Oct-2022.
  Fact 0.1: Sambhaji Raje's field is Social work and affiliation is Bharatiya Janata Party, with appointment on 07-Jun-2016 and retirement on 03-May-2022.
  Fact 0.2: Suresh Gopi's field is Art and affiliation is Bharatiya Janata Party, with appointment on 25-Apr-2016 and retirement on 24-Apr-2022.
  Fact 0.3: Subramanian Swamy's field is Economics and affiliation is Bharatiya Janata Party, with appointment on 25-Apr-2016 and retirement on 24-Apr-2022.
  Fact 0.4: Narendra Jadhav (No. 5) has field Economics and affiliation Nominated, with appointment on 25-Apr-2016 and retirement on 24-Apr-2022.
  Fact 0.5: Mary Kom (No. 6) has field Sport and affiliation Nominated, with appointment on 25-Apr-2016 and retirement on 24-Apr-2022.
  Fact 0.6: Swapan Dasgupta (No. 7) has field Journalism and affiliation Nominated, with appointment on 25-Apr-2016 and retirement on 24-Apr-2022.
  Fact 0.7: K.T.S. Tulsi (No. 8) has field Law and affiliation Nominated, with appointment on 25-Feb-2014 and retirement on 24-Feb-2020.
  Fact 0.8: K. Parasaran (No. 9) has field Law and affiliation Nominated, with appointment on 09-Jun-2012 and retirement on 28-Jun-2018.
  Fact 0.9: Rekha (No. 10) has field Art and affiliation Nominated, with appointment on 27-Apr-2012 and retirement on 26-Apr-2018.
  Fact 0.10: Sachin Tendulkar (No. 11) has field Social service and affiliation Nominated, with appointment on 27-Apr-2012 and retirement on 26-Apr-2018.
  Fact 0.11: Anu Aga (No. 12) has field Business and affiliation Nominated, with appointment on 27-Apr-2012 and retirement on 26-Apr-2018.

Retrieved items:
<item 1>
List of nominated members of Rajya Sabha
Krishnamachari 1961 -- 1967 24 -- R.R. Diwakar 1962 -- 1968 25 -- Gopal Singh 1962 -- 1968 26 -- M. Ajmal Khan 1964 -- 1972 27 -- Satyavrata Siddhantalankar
1964 -- 1968 28 Shakuntala Paranjpye 1964 -- 1970 29 -- Badri Nath Prasad 1964 -- 1970 30 -- G. Ramachandran 1964 -- 1970 31 -- M.N. Kaul 1964 -- 1972 32 --
Harivanshrai Bachchan 1966 -- 1972 33 -- Dhananjay Ramchandra Gadgil 1966 -- 1972 34 -- M.C.
</item 1>

<item 2>
List of nominated members of Rajya Sabha
Narayana Reddy 1997 -- 2003 93 Mrinal Sen 1997 -- 2003 94 -- Chaudhary Harmohan Singh Yadav 1997 -- 2003 95 -- Nanaji Deshmukh 1999 -- 2005 96 Lata Mangeshkar
1999 -- 2005 97 -- Fali Nariman 1999 -- 2005 98 -- Cho Ramaswamy 1999 -- 2005 99 Hema Malini 2003 -- 2009 100 -- Bimal Jalan 2003 -- 2009 101 K.
</item 2>

<item 3>
List of nominated members of Rajya Sabha
Tulsi Law Nominated 25 - Feb - 2014 24 - Feb - 2020 9 K. Parasaran Law Nominated 09 - Jun - 2012 28 - Jun - 2018 10 Rekha Art Nominated 27 - Apr - 2012 26 - Apr
- 2018 11 Sachin Tendulkar Social service Nominated 27 - Apr - 2012 26 - Apr - 2018 12 Anu Aga Business Nominated 27 - Apr - 2012 26 - Apr - 2018 Complete list
of nominated members of Rajya Sabha ( edit ) This is a Complete list of Members of the Rajya Sabha who have been nominated by the President . Sr. No .
</item 3>

<item 4>
List of nominated members of Rajya Sabha
Dutta 1993 -- 1999 85 -- Maulana Habibur Rahman Nomani 1993 -- 1999 86 Nirmala Deshpande 1997 -- 1999 and 2004 -- 2010 87 Shabana Azmi 1997 -- 2003 88 -- P.
Selvie Das 1997 -- 2003 89 -- Kartar Singh Duggal 1997 -- 2003 90 Kuldip Nayyar 1997 -- 2003 91 -- Raja Ramanna 1997 -- 2003 92 C.
</item 4>

<item 5>
List of nominated members of Rajya Sabha
No . Name Field Affiliation Date of Appointment Date of Retirement Roopa Ganguly Art Bharatiya Janata Party 04 - Oct - 2016 03 - Oct - 2022 Sambhaji Raje Social
work Bharatiya Janata Party 07 - Jun - 2016 03 - May - 2022 Suresh Gopi Art Bharatiya Janata Party 25 - Apr - 2016 24 - Apr - 2022 Subramanian Swamy Economics
Bharatiya Janata Party 25 - Apr - 2016 24 - Apr - 2022 5 Narendra Jadhav Economics Nominated 25 - Apr - 2016 24 - Apr - 2022 6 Mary Kom Sport Nominated 25 - Apr
- 2016 24 - Apr - 2022 7 Swapan Dasgupta Journalism Nominated 25 - Apr - 2016 24 - Apr - 2022 8 K.T.S.
</item 5>

<item 6>
List of nominated members of Rajya Sabha
Daphtary 1972 -- 1978 47 -- Habib Tanvir 1972 -- 1978 48 -- Krishna Kriplani 1974 -- 1980 49 Lokesh Chandra 1974 -- 1986 50 -- Scato Swu 1974 -- 1986 51 -- B.N.
Banerjee 1976 -- 1982 52 -- Malcolm Adiseshiah 1978 -- 1984 53 -- Fathema Ismail 1978 -- 1984 54 -- Pandurang Dharmaji Jadhav 1978 -- 1984 55 -- Bhagwati Charan
Varma 1978 -- 1984 56 Nargis Dutt 1980 -- 1986 57 Khushwant Singh 1980 -- 1986 58 -- Asima Chatterjee 1982 -- 1990 59 Sivaji Ganesan 1982 -- 1986 60 --
Hayatullah Ansari 1982 -- 1988 61 -- Madan Bhatia 1982 -- 1994 62 -- V.N.
</item 6>

<item 7>
Graph query row:
person: Rukmini Devi Arundale
occupation: None
relationship: MEMBER_OF
connected_to: Rajya Sabha
</item 7>

<item 8>
List of nominated members of Rajya Sabha
Tulsi 2014 - 2020 125 Subramanian Swamy 2016 - 2022 126 MC Mary Kom 2016 - 2022 127 Narendra Jadhav 2016 - 2022 129 Suresh Gopi 2016 - 2022 130 -- Swapan
Dasgupta 2016 - 2022 131 -- Sambhaji Raje 2016 - 2022 132 -- Roopa Ganguly 2016 - 2022 References ( edit ) Nominated Members since 1952 Narender Kumar Current
list of Nominated Members Specific Jump up ^ `` Statewise List '' . 164.100. 47.5 . Retrieved 12 June 2016 . Jump up ^
http://164.100.47.5/Newmembers/partymemberlist.aspx ^ Jump up to : `` Statewise Retirement '' . 164.100. 47.5 . Retrieved 12 June 2016 .
</item 8>

<item 9>
List of nominated members of Rajya Sabha
Kasturirangan 2003 -- 2009 102 -- Narayan Singh Manaklao 2003 -- 2009 103 -- Vidya Niwas Misra 2003 -- 2009 104 -- Chandan Mitra 2003 -- 2009 105 Dara Singh
2003 -- 2009 106 Kapila Vatsyayan 2006 -- 2012 107 Ram Jethmalani 2006 -- 2009 108 -- Shobhana Bhartia 2006 -- 2012 109 Shyam Benegal 2006 -- 2012 110 M.S.
Swaminathan 2007 -- 2013 111 -- C. Rangarajan 2008 -- 2010 112 -- H.K.
</item 9>

<item 10>
List of nominated members of Rajya Sabha
Sr. No . Image Name Tenure -- Alladi Krishnaswamy Iyer 1952 -- 1953 Satyendranath Bose 1952 -- 1960 Prithviraj Kapoor 1952 -- 1960 -- Jagadisan Mohandas
Kumarappa 1952 -- 1954 5 -- Kalidas Nag 1952 -- 1954 6 Rukmini Devi Arundale 1952 -- 1962 7 -- N.R. Malkani 1952 -- 1962 8 -- Sahib Singh Sokhey 1952 -- 1956 9
-- Zakir Hussain 1952 -- 1962 10 -- Maithili Sharan Gupt 1952 -- 1964 11 -- Kakasaheb Kalelkar 1952 -- 1964 12 -- Radha Kumud Mukherjee 1952 -- 1958 13 P.V.
</item 10>
=== END ITEM id=664fd91dc7152e0c ===

=== ITEM id=67cfb79b818c7c39 ===
Gold sentences and facts:
Sentence 0: The book became an instant hit , and the online version received about 20 million views as of 2009 ; many online readers requested a printed version . Kinney had agreed , and in April 2007 , Diary of a Wimpy Kid was published . To date , twelve Wimpy Kid books have been released , plus one do - it - yourself book and two movie diaries . In April 2009 , TIME named Kinney one of The World 's Most Influential People .
  Fact 0.0: The book became an instant hit.
  Fact 0.1: The online version of the book received about 20 million views as of 2009.
  Fact 0.2: Many online readers requested a printed version of the book.
  Fact 0.3: Kinney agreed to a printed version of the book.
  Fact 0.4: Diary of a Wimpy Kid was published in April 2007.
  Fact 0.5: To date, twelve Wimpy Kid books have been released.
  Fact 0.6: To date, one do-it-yourself Wimpy Kid book has been released.
  Fact 0.7: To date, two Wimpy Kid movie diaries have been released.
  Fact 0.8: In April 2009, TIME named Kinney one of The World's Most Influential People.
Sentence 1: # Title Date ISBN Diary of a Wimpy Kid April 1 , 2007 ISBN 978 - 0 - 8109 - 9313 - 6 The introduction to the series introduces the protagonist , Greg Heffley , his best friend Rowley , his middle school problems , Halloween , and the Cheese Touch . Rodrick Rules February 1 , 2008 ISBN 978 - 0 - 8109 - 9473 - 7 Greg must stop Rodrick from revealing the secrets that had happened last summer . The Last Straw January 13 , 2009 ISBN 978 - 0 - 8109 - 7068 - 7 When Greg 's father threatens to send Greg to a military academy , Greg must find a way to prevent this from happening . Dog Days October 12 , 2009 ISBN 978 - 0 - 8109 - 8391 - 5 Summer is here , but when the family has not that much money to go to the beach ( money is tight at that time ) , it might not be the best summer ever . 5 The Ugly Truth November 9 , 2010 ISBN 978 - 0 - 8109 - 8491 - 2 Greg is growing up , and he will have to face the challenges of this period ( like changing dentists , and a school lock - in ) . 6 Cabin Fever November 15 , 2011 ISBN 978 - 1 - 4197 - 0223 - 5 The family is stranded in the house as snow begins to fall . 7 The Third Wheel November 13 , 2012 ISBN 978 - 1 - 4197 - 0584 - 7 The Valentine 's Day dance is coming up , and Greg has no one go with . But when Michael Sampson ( Abigail 's date ) will not be going , things turn around . 8 Hard Luck November 5 , 2013 ISBN 978 - 1 - 4197 - 1132 - 9 Greg 's friend count is dropping as Rowley now has a girlfriend , named Abigail ( continuing the last book ) . 9 The Long Haul November 4 , 2014 ISBN 978 - 1 - 4197 - 1189 - 3 This is the family 's second summer trip , but instead , it 's a road trip . But with the Heffleys , you know what to expect from this disaster . 10 Old School November 3 , 2015 ISBN 978 - 1 - 4197 - 1701 - 7 The neighborhood is going to cut electricity for the weekend , and as the family 's car has a fender bender , it forces Greg to go to the farm trip . Will Greg survive the electricity - free days ? 11 Double Down November 1 , 2016 ISBN 978 - 1 - 4197 - 2344 - 5 Greg 's mom wants him to do something else than video games , so he makes a movie in collaboration with Rowley and joins the music program . 12 The Getaway November 7 , 2017 ISBN 978 - 1 - 4197 - 2545 - 6 Instead of celebrating the holidays at home , Greg and his family head on a tropical vacation to a resort . But what first starts as an relaxing trip , later turns into a Christmas nightmare . Will Greg be able to turn this vacation around ?
  Fact 1.0: Diary of a Wimpy Kid was released on April 1, 2007, with ISBN 978-0-8109-9313-6.
  Fact 1.1: The introduction to the Diary of a Wimpy Kid series introduces the protagonist Greg Heffley, his best friend Rowley, his middle school problems, Halloween, and the Cheese Touch.
  Fact 1.2: Rodrick Rules was released on February 1, 2008, with ISBN 978-0-8109-9473-7.
  Fact 1.3: In Rodrick Rules, Greg must stop Rodrick from revealing the secrets that happened last summer.
  Fact 1.4: The Last Straw was released on January 13, 2009, with ISBN 978-0-8109-7068-7.
  Fact 1.5: In The Last Straw, when Greg's father threatens to send Greg to a military academy, Greg must find a way to prevent this.
  Fact 1.6: Dog Days was released on October 12, 2009, with ISBN 978-0-8109-8391-5.
  Fact 1.7: In Dog Days, summer is here, but the family does not have much money to go to the beach, so it might not be the best summer ever.
  Fact 1.8: The Ugly Truth was released on November 9, 2010, with ISBN 978-0-8109-8491-2.
  Fact 1.9: In The Ugly Truth, Greg is growing up and will have to face the challenges of this period, like changing dentists and a school lock-in.
  Fact 1.10: Cabin Fever was released on November 15, 2011, with ISBN 978-1-4197-0223-5.
  Fact 1.11: In Cabin Fever, the family is stranded in the house as snow begins to fall.
  Fact 1.12: The Third Wheel was released on November 13, 2012, with ISBN 978-1-4197-0584-7.
  Fact 1.13: In The Third Wheel, the Valentine's Day dance is coming up and Greg has no one to go with.
  Fact 1.14: In The Third Wheel, things turn around when Michael Sampson, Abigail's date, will not be going.
  Fact 1.15: Hard Luck was released on November 5, 2013, with ISBN 978-1-4197-1132-9.
  Fact 1.16: In Hard Luck, Greg's friend count is dropping as Rowley now has a girlfriend, named Abigail.
  Fact 1.17: Hard Luck continues the story from The Third Wheel.
  Fact 1.18: The Long Haul was released on November 4, 2014, with ISBN 978-1-4197-1189-3.
  Fact 1.19: In The Long Haul, this is the family's second summer trip, and it is a road trip.
  Fact 1.20: Old School was released on November 3, 2015, with ISBN 978-1-4197-1701-7.
  Fact 1.21: In Old School, the neighborhood is going to cut electricity for the weekend.
  Fact 1.22: In Old School, the family's car has a fender bender, which forces Greg to go on the farm trip.
  Fact 1.23: Old School asks whether Greg will survive the electricity-free days.
  Fact 1.24: Double Down was released on November 1, 2016, with ISBN 978-1-4197-2344-5.
  Fact 1.25: In Double Down, Greg's mom wants him to do something other than video games, so he makes a movie in collaboration with Rowley.
  Fact 1.26: In Double Down, Greg joins the music program.
  Fact 1.27: The Getaway was released on November 7, 2017, with ISBN 978-1-4197-2545-6.
  Fact 1.28: In The Getaway, instead of celebrating the holidays at home, Greg and his family head on a tropical vacation to a resort.
  Fact 1.29: In The Getaway, what first starts as a relaxing trip later turns into a Christmas nightmare.
  Fact 1.30: The Getaway asks whether Greg will be able to turn the vacation around.
Sentence 2: Diary of a Wimpy Kid The logo of the series , appearing on the book covers , blogs , web comic , and the films . Diary of a Wimpy Kid Diary of a Wimpy Kid : Rodrick Rules Diary of a Wimpy Kid : The Last Straw Diary of a Wimpy Kid : Dog Days Diary of a Wimpy Kid : The Ugly Truth Diary of a Wimpy Kid : Cabin Fever Diary of a Wimpy Kid : The Third Wheel Diary of a Wimpy Kid : Hard Luck Diary of a Wimpy Kid : The Long Haul Diary of a Wimpy Kid : Old School Diary of a Wimpy Kid : Double Down Diary of a Wimpy Kid : The Getaway Author Jeff Kinney Country United States Language English Genre Realistic fiction Publisher Scholastic Publishing / Amulet Books / Puffin Books Published April 1 , 2007 -- present Media type Print ( hardback & paperback ) Audiobook E-book
  Fact 2.0: The logo of the Diary of a Wimpy Kid series appears on the book covers, blogs, web comic, and films.
  Fact 2.1: The Diary of a Wimpy Kid series includes the books Rodrick Rules, The Last Straw, Dog Days, The Ugly Truth, Cabin Fever, The Third Wheel, Hard Luck, The Long Haul, Old School, Double Down, and The Getaway.
  Fact 2.2: The Diary of a Wimpy Kid series was written by Jeff Kinney.
  Fact 2.3: The Diary of a Wimpy Kid series is from the United States.
  Fact 2.4: The Diary of a Wimpy Kid series is in English.
  Fact 2.5: The Diary of a Wimpy Kid series is in the realistic fiction genre.
  Fact 2.6: The Diary of a Wimpy Kid series is published by Scholastic Publishing, Amulet Books, and Puffin Books.
  Fact 2.7: The Diary of a Wimpy Kid series was published from April 1, 2007 to the present.
  Fact 2.8: The Diary of a Wimpy Kid series is available in print (hardback and paperback), as an audiobook, and as an e-book.

Retrieved items:
<item 1>
Diary of a Wimpy Kid (book series)
USA Today , which ranks best sellers based on sales alone regardless of genre or intended audience , has listed Diary of a Wimpy Kid in its top 150 , peaking at
number 8 , for 141 weeks as of April 4 , 2010 . As of the same date , Rodrick Rules has been on the list for 117 weeks ( peaking at number 4 ) , The Last Straw
has been on the list for 65 weeks ( peaking at number 1 ) , Dog Days has been on the list for all 25 weeks of its publication ( peaking at number 1 ) , and The
Wimpy Kid Movie Diary has been listed for all three weeks of its publication , peaking at number 2 . The series has sold over 194 million books in more than 30
countries . The series has also generated over $500 million in revenue . Films ( edit ) Main article : Diary of a Wimpy Kid ( film series ) This section needs
to be updated . Please update this article to reflect recent events or newly available information .
</item 1>

<item 2>
Diary of a Wimpy Kid (book series)
ISBN 0 - 8109 - 9313 - 9 . ^ Jump up to : `` '' Diary of a Wimpy Kid ( 2010 ) '' on `` . Comingsoon.net . March 19 , 2010 . Retrieved May 7 , 2011 . ^ Jump up
to : `` Diary of a Wimpy Kid '' . FunBrain . Family Education Network . Retrieved March 5 , 2011 . Jump up ^ `` WIMPY 2 LAST '' . 3 November 2015 . Retrieved 6
March 2016 . Jump up ^ The 2009 TIME 100 : Jeff Kinney Time magazine 's 2009 list of The World 's Most Influential People Jump up ^ Kinney , Jeff . `` The Wimpy
Kid Movie Diary : The Next Chapter '' . Retrieved March 28 , 2017 . Jump up ^ Nickelodeon Kids ' Choice Awards , Nick.com Jump up ^ Dixler , Elsa . `` The New
York Times Best Sellers : Children 's Books '' . The New York Times .
</item 2>

<item 3>
Diary of a Wimpy Kid (book series)
Contents ( hide ) 1 Background 2 Chronology 2.1 Supplementary books 2.2 Toys and games 2.3 Audiobooks 3 Characters 4 Achievements 4.1 Awards and honors 4.2
Commercial success 5 Films 6 Musical 7 References 8 External links Background ( edit ) In January 1998 , Diary of a Wimpy Kid officially began , when Kinney
brought up an idea about a middle - school weakling named Greg Heffley , who wrote about his personal life . In May 2004 , FunBrain and Kinney released an
online version of Diary of a Wimpy Kid . The website made daily entries until June 2005 . He worked on his book for almost eight years , before showing it to a
publisher in New York . In February 2006 , during the New York Comic Con , Jeff signed a multi book deal with publisher Harry N. Abrams , Inc. , to turn Diary
of a Wimpy Kid into a print series .
</item 3>

<item 4>
Diary of a Wimpy Kid (book series)
Will Greg be able to turn this vacation around ? Supplementary books ( edit ) Diary of a Wimpy Kid Do - It - Yourself Book ( published between Rodrick Rules and
The Last Straw ) is an activity book that features a blank section encouraging readers to keep their own journal in the style of the books . The Wimpy Kid Do -
It - Yourself Book : Extended Edition ( published after Wimpy Kid Movie Diary 2 and before Cabin Fever ) is the same book as the first Do - It - Yourself Book ,
but with extra 60 pages and 16 more full - color comics . The Wimpy Kid Movie Diary is a book about the making of the first film , which features stills and
brand - new illustrations . Jeff Kinney dedicated the book to the film 's main stars , Zachary Gordon and Robert Capron . The Wimpy Kid Movie Diary 2 : Rodrick
Rules is an updated version of the first book , now featuring stills from the second film Rodrick Rules , more new illustrations , and information on how the
second film was made .
</item 4>

<item 5>
Diary of a Wimpy Kid (book series)
Diary of a Wimpy Kid Diary of a Wimpy Kid : Rodrick Rules Diary of a Wimpy Kid : The Last Straw Diary of a Wimpy Kid : Dog Days Diary of a Wimpy Kid : The Ugly
Truth Diary of a Wimpy Kid : Cabin Fever Diary of a Wimpy Kid : The Third Wheel Diary of a Wimpy Kid : Hard Luck Diary of a Wimpy Kid : The Long Haul Diary of a
Wimpy Kid : Old School Diary of a Wimpy Kid : Double Down Diary of a Wimpy Kid : The Getaway Author Jeff Kinney Country United States Language English Genre
Realistic fiction Publisher Scholastic Publishing / Amulet Books / Puffin Books Published April 1 , 2007 -- present Media type Print ( hardback & paperback )
Audiobook E-book Diary of a Wimpy Kid is a series of fiction books written by the American author and cartoonist Jeff Kinney . All the main books are the
journals of the main character , Greg Heffley .
</item 5>

<item 6>
Diary of a Wimpy Kid (book series)
All the main books are the journals of the main character , Greg Heffley . Befitting a teen 's diary , the books are filled with hand - written notes and simple
drawings of Greg 's daily adventures . Since the release of the online version in May 2004 , most of the books have garnered positive reviews and commercial
success . As of February 2008 , it has been purchased one million times . The first , second , fourth , and ninth installments have been adapted into films by
20th Century Fox . The website Poptropica released islands in the theme of Diary of a Wimpy Kid called Wimpy Wonderland and Wimpy Boardwalk , both of which
credit Kinney in an ambiguous contributor 's role .
</item 6>

<item 7>
Diary of a Wimpy Kid (book series)
Diary of a Wimpy Kid ( book series ) - Wikipedia Diary of a Wimpy Kid ( book series ) This is the latest accepted revision , reviewed on 22 April 2018 . Jump to
: navigation , search `` Diary of a Wimpy Kid ( series ) '' redirects here . For the film series , see Diary of a Wimpy Kid ( film series ) . Diary of a Wimpy
Kid The logo of the series , appearing on the book covers , blogs , web comic , and the films .
</item 7>

<item 8>
Diary of a Wimpy Kid (book series)
The Wimpy Kid Movie Diary 3 : Dog Days is the second updated version of the first book , now adding stills from the third film Dog Days , and information on how
this film was made . The Wimpy Kid Movie Diary : The Next Chapter covers how the film The Long Haul was made . It is dedicated to Jason Drucker , the actor who
played Greg in that movie . Toys and games ( edit ) Action figures included a minifigure pack ( with Greg , Rowley , and Fregley ) , a 6 - inch Greg Heffley
figure , a 6 - inch holiday figure based on Cabin Fever , and a Greg and Rodrick plush doll . Audiobooks ( edit ) Twelve audiobooks have been released for the
Diary of a Wimpy Kid series . The first six are read by Ramon de Ocampo and Dan Russell from number seven . They are presented by Recorded Books and distributed
by Audible Inc . Diary of a Wimpy Kid ( April 1 , 2007 ) Diary of a Wimpy Kid : Rodrick Rules ( February 1 ,
</item 8>

<item 9>
Diary of a Wimpy Kid (book series)
2007 ) Diary of a Wimpy Kid : Rodrick Rules ( February 1 , 2008 ) Diary of a Wimpy Kid : The Last Straw ( January 13 , 2009 ) Diary of a Wimpy Kid : Dog Days (
October 12 , 2009 ) Diary of a Wimpy Kid : The Ugly Truth ( November 9 , 2010 ) Diary of a Wimpy Kid : Cabin Fever ( November 15 , 2011 ) Diary of a Wimpy Kid :
The Third Wheel ( November 13 , 2012 ) Diary of a Wimpy Kid : Hard Luck ( November 5 , 2013 ) Diary of a Wimpy Kid : The Long Haul ( November 4 , 2014 ) Diary
of a Wimpy Kid : Old School ( November 3 , 2015 ) Diary of a Wimpy Kid : Double Down ( November 1 , 2016 ) Diary of a Wimpy Kid : The Getaway ( November 7 ,
</item 9>

<item 10>
Diary of a Wimpy Kid (book series)
, to turn Diary of a Wimpy Kid into a print series . The book became an instant hit , and the online version received about 20 million views as of 2009 ; many
online readers requested a printed version . Kinney had agreed , and in April 2007 , Diary of a Wimpy Kid was published . To date , twelve Wimpy Kid books have
been released , plus one do - it - yourself book and two movie diaries . In April 2009 , TIME named Kinney one of The World 's Most Influential People .
Chronology ( edit ) # Title Date ISBN Diary of a Wimpy Kid April 1 , 2007 ISBN 978 - 0 - 8109 - 9313 - 6 The introduction to the series introduces the
protagonist , Greg Heffley , his best friend Rowley , his middle school problems , Halloween , and the Cheese Touch . Rodrick Rules February 1 , 2008 ISBN 978 -
0 - 8109 - 9473 - 7 Greg must stop Rodrick from revealing the secrets that had happened last summer .
</item 10>
=== END ITEM id=67cfb79b818c7c39 ===

=== ITEM id=5a550affcf5fc5ae ===
Gold sentences and facts:
Sentence 0: No . Name Field Affiliation Date of Appointment Date of Retirement Roopa Ganguly Art Bharatiya Janata Party 04 - Oct - 2016 03 - Oct - 2022 Sambhaji Raje Social work Bharatiya Janata Party 07 - Jun - 2016 03 - May - 2022 Suresh Gopi Art Bharatiya Janata Party 25 - Apr - 2016 24 - Apr - 2022 Subramanian Swamy Economics Bharatiya Janata Party 25 - Apr - 2016 24 - Apr - 2022 5 Narendra Jadhav Economics Nominated 25 - Apr - 2016 24 - Apr - 2022 6 Mary Kom Sport Nominated 25 - Apr - 2016 24 - Apr - 2022 7 Swapan Dasgupta Journalism Nominated 25 - Apr - 2016 24 - Apr - 2022 8 K.T.S. Tulsi Law Nominated 25 - Feb - 2014 24 - Feb - 2020 9 K. Parasaran Law Nominated 09 - Jun - 2012 28 - Jun - 2018 10 Rekha Art Nominated 27 - Apr - 2012 26 - Apr - 2018 11 Sachin Tendulkar Social service Nominated 27 - Apr - 2012 26 - Apr - 2018 12 Anu Aga Business Nominated 27 - Apr - 2012 26 - Apr - 2018
  Fact 0.0: Roopa Ganguly's field is Art and affiliation is Bharatiya Janata Party, with appointment on 04-Oct-2016 and retirement on 03-Oct-2022.
  Fact 0.1: Sambhaji Raje's field is Social work and affiliation is Bharatiya Janata Party, with appointment on 07-Jun-2016 and retirement on 03-May-2022.
  Fact 0.2: Suresh Gopi's field is Art and affiliation is Bharatiya Janata Party, with appointment on 25-Apr-2016 and retirement on 24-Apr-2022.
  Fact 0.3: Subramanian Swamy's field is Economics and affiliation is Bharatiya Janata Party, with appointment on 25-Apr-2016 and retirement on 24-Apr-2022.
  Fact 0.4: Narendra Jadhav (No. 5) has field Economics and affiliation Nominated, with appointment on 25-Apr-2016 and retirement on 24-Apr-2022.
  Fact 0.5: Mary Kom (No. 6) has field Sport and affiliation Nominated, with appointment on 25-Apr-2016 and retirement on 24-Apr-2022.
  Fact 0.6: Swapan Dasgupta (No. 7) has field Journalism and affiliation Nominated, with appointment on 25-Apr-2016 and retirement on 24-Apr-2022.
  Fact 0.7: K.T.S. Tulsi (No. 8) has field Law and affiliation Nominated, with appointment on 25-Feb-2014 and retirement on 24-Feb-2020.
  Fact 0.8: K. Parasaran (No. 9) has field Law and affiliation Nominated, with appointment on 09-Jun-2012 and retirement on 28-Jun-2018.
  Fact 0.9: Rekha (No. 10) has field Art and affiliation Nominated, with appointment on 27-Apr-2012 and retirement on 26-Apr-2018.
  Fact 0.10: Sachin Tendulkar (No. 11) has field Social service and affiliation Nominated, with appointment on 27-Apr-2012 and retirement on 26-Apr-2018.
  Fact 0.11: Anu Aga (No. 12) has field Business and affiliation Nominated, with appointment on 27-Apr-2012 and retirement on 26-Apr-2018.

Retrieved items:
<item 1>
List of nominated members of Rajya Sabha
No . Name Field Affiliation Date of Appointment Date of Retirement Roopa Ganguly Art Bharatiya Janata Party 04 - Oct - 2016 03 - Oct - 2022 Sambhaji Raje Social
work Bharatiya Janata Party 07 - Jun - 2016 03 - May - 2022 Suresh Gopi Art Bharatiya Janata Party 25 - Apr - 2016 24 - Apr - 2022 Subramanian Swamy Economics
Bharatiya Janata Party 25 - Apr - 2016 24 - Apr - 2022 5 Narendra Jadhav Economics Nominated 25 - Apr - 2016 24 - Apr - 2022 6 Mary Kom Sport Nominated 25 - Apr
- 2016 24 - Apr - 2022 7 Swapan Dasgupta Journalism Nominated 25 - Apr - 2016 24 - Apr - 2022 8 K.T.S.
</item 1>

<item 2>
List of nominated members of Rajya Sabha
Dutta 1993 -- 1999 85 -- Maulana Habibur Rahman Nomani 1993 -- 1999 86 Nirmala Deshpande 1997 -- 1999 and 2004 -- 2010 87 Shabana Azmi 1997 -- 2003 88 -- P.
Selvie Das 1997 -- 2003 89 -- Kartar Singh Duggal 1997 -- 2003 90 Kuldip Nayyar 1997 -- 2003 91 -- Raja Ramanna 1997 -- 2003 92 C.
</item 2>

<item 3>
List of nominated members of Rajya Sabha
Tulsi 2014 - 2020 125 Subramanian Swamy 2016 - 2022 126 MC Mary Kom 2016 - 2022 127 Narendra Jadhav 2016 - 2022 129 Suresh Gopi 2016 - 2022 130 -- Swapan
Dasgupta 2016 - 2022 131 -- Sambhaji Raje 2016 - 2022 132 -- Roopa Ganguly 2016 - 2022 References ( edit ) Nominated Members since 1952 Narender Kumar Current
list of Nominated Members Specific Jump up ^ `` Statewise List '' . 164.100. 47.5 . Retrieved 12 June 2016 . Jump up ^
http://164.100.47.5/Newmembers/partymemberlist.aspx ^ Jump up to : `` Statewise Retirement '' . 164.100. 47.5 . Retrieved 12 June 2016 .
</item 3>

<item 4>
List of nominated members of Rajya Sabha
Krishnamachari 1961 -- 1967 24 -- R.R. Diwakar 1962 -- 1968 25 -- Gopal Singh 1962 -- 1968 26 -- M. Ajmal Khan 1964 -- 1972 27 -- Satyavrata Siddhantalankar
1964 -- 1968 28 Shakuntala Paranjpye 1964 -- 1970 29 -- Badri Nath Prasad 1964 -- 1970 30 -- G. Ramachandran 1964 -- 1970 31 -- M.N. Kaul 1964 -- 1972 32 --
Harivanshrai Bachchan 1966 -- 1972 33 -- Dhananjay Ramchandra Gadgil 1966 -- 1972 34 -- M.C.
</item 4>

<item 5>
List of nominated members of Rajya Sabha
Tiwari 1982 -- 1984 63 -- Ghulam Rasool Kar 1984 -- 1987 64 -- Tindivanam K. Ramamurthy 1984 -- 1990 65 -- H.L. Kapur 1985 -- 1988 66 Salim Ali 1985 -- 1988 67
-- Purushottam Kakodkar 1985 -- 1991 68 Ela Bhatt 1986 -- 1988 69 -- Amrita Pritam 1986 -- 1992 70 M.F. Husain 1986 -- 1992 71 R.K.
</item 5>

<item 6>
List of nominated members of Rajya Sabha
Kasturirangan 2003 -- 2009 102 -- Narayan Singh Manaklao 2003 -- 2009 103 -- Vidya Niwas Misra 2003 -- 2009 104 -- Chandan Mitra 2003 -- 2009 105 Dara Singh
2003 -- 2009 106 Kapila Vatsyayan 2006 -- 2012 107 Ram Jethmalani 2006 -- 2009 108 -- Shobhana Bhartia 2006 -- 2012 109 Shyam Benegal 2006 -- 2012 110 M.S.
Swaminathan 2007 -- 2013 111 -- C. Rangarajan 2008 -- 2010 112 -- H.K.
</item 6>

<item 7>
List of nominated members of Rajya Sabha
Narayana Reddy 1997 -- 2003 93 Mrinal Sen 1997 -- 2003 94 -- Chaudhary Harmohan Singh Yadav 1997 -- 2003 95 -- Nanaji Deshmukh 1999 -- 2005 96 Lata Mangeshkar
1999 -- 2005 97 -- Fali Nariman 1999 -- 2005 98 -- Cho Ramaswamy 1999 -- 2005 99 Hema Malini 2003 -- 2009 100 -- Bimal Jalan 2003 -- 2009 101 K.
</item 7>

<item 8>
List of nominated members of Rajya Sabha
Daphtary 1972 -- 1978 47 -- Habib Tanvir 1972 -- 1978 48 -- Krishna Kriplani 1974 -- 1980 49 Lokesh Chandra 1974 -- 1986 50 -- Scato Swu 1974 -- 1986 51 -- B.N.
Banerjee 1976 -- 1982 52 -- Malcolm Adiseshiah 1978 -- 1984 53 -- Fathema Ismail 1978 -- 1984 54 -- Pandurang Dharmaji Jadhav 1978 -- 1984 55 -- Bhagwati Charan
Varma 1978 -- 1984 56 Nargis Dutt 1980 -- 1986 57 Khushwant Singh 1980 -- 1986 58 -- Asima Chatterjee 1982 -- 1990 59 Sivaji Ganesan 1982 -- 1986 60 --
Hayatullah Ansari 1982 -- 1988 61 -- Madan Bhatia 1982 -- 1994 62 -- V.N.
</item 8>

<item 9>
List of nominated members of Rajya Sabha
Husain 1986 -- 1992 71 R.K. Narayan 1986 -- 1992 72 Ravi Shankar 1986 -- 1992 73 -- Syeda Anwara Taimur 1988 -- 1990 74 -- Sat Paul Mittal 1988 -- 1994 75 --
Bishambhar Nath Pande 1988 -- 1994 76 -- Mohammad Yunus 1989 -- 1995 77 Prakash Ambedkar 1990 -- 1996 78 Jagmohan 1990 -- 1996 79 -- Bhupinder Singh Mann 1990
-- 1996 80 -- Russi Karanjia 1991 -- 1997 81 -- Mahendra Prasad 1993 -- 1994 82 -- M. Aram 1993 -- 1999 83 Vyjayantimala Bali 1993 -- 1999 84 -- B.B.
</item 9>

<item 10>
List of nominated members of Rajya Sabha
Sr. No . Image Name Tenure -- Alladi Krishnaswamy Iyer 1952 -- 1953 Satyendranath Bose 1952 -- 1960 Prithviraj Kapoor 1952 -- 1960 -- Jagadisan Mohandas
Kumarappa 1952 -- 1954 5 -- Kalidas Nag 1952 -- 1954 6 Rukmini Devi Arundale 1952 -- 1962 7 -- N.R. Malkani 1952 -- 1962 8 -- Sahib Singh Sokhey 1952 -- 1956 9
-- Zakir Hussain 1952 -- 1962 10 -- Maithili Sharan Gupt 1952 -- 1964 11 -- Kakasaheb Kalelkar 1952 -- 1964 12 -- Radha Kumud Mukherjee 1952 -- 1958 13 P.V.
</item 10>
=== END ITEM id=5a550affcf5fc5ae ===

=== ITEM id=1413d5627b578f45 ===
Gold sentences and facts:
Sentence 0: Kylo Ren is a fictional character in the Star Wars franchise . Introduced in the 2015 film Star Wars : The Force Awakens , he is portrayed by Adam Driver . Kylo Ren is the chosen name of Ben Solo , the son of original Star Wars trilogy characters Han Solo and Leia Organa . Though trained by his uncle Luke Skywalker as a Jedi , he has been seduced to the dark side of the Force by Supreme Leader Snoke and aspires to be as powerful as his grandfather , Darth Vader . Kylo Ren is also the master of the Knights of Ren , as well as a commander and later the supreme leader of the First Order , an organization spawned from the fallen Galactic Empire . He is featured in The Force Awakens media and merchandising and appears in the film 's sequel , Star Wars : The Last Jedi ( 2017 ) .
  Fact 0.0: Kylo Ren is a fictional character in the Star Wars franchise.
  Fact 0.1: Kylo Ren was introduced in the 2015 film Star Wars: The Force Awakens.
  Fact 0.2: Kylo Ren is portrayed by Adam Driver.
  Fact 0.3: Kylo Ren is the chosen name of Ben Solo.
  Fact 0.4: Ben Solo is the son of original Star Wars trilogy characters Han Solo and Leia Organa.
  Fact 0.5: Kylo Ren was trained by his uncle Luke Skywalker as a Jedi.
  Fact 0.6: Kylo Ren has been seduced to the dark side of the Force by Supreme Leader Snoke.
  Fact 0.7: Kylo Ren aspires to be as powerful as his grandfather, Darth Vader.
  Fact 0.8: Kylo Ren is the master of the Knights of Ren.
  Fact 0.9: Kylo Ren is a commander of the First Order.
  Fact 0.10: Kylo Ren later became the supreme leader of the First Order.
  Fact 0.11: The First Order is an organization spawned from the fallen Galactic Empire.
  Fact 0.12: Kylo Ren is featured in The Force Awakens media and merchandising.
  Fact 0.13: Kylo Ren appears in the film's sequel, Star Wars: The Last Jedi.
  Fact 0.14: Star Wars: The Last Jedi was released in 2017.
Sentence 1: Abrams told Empire in August 2015 , `` Kylo Ren is not a Sith . He works under Supreme Leader Snoke , who is a powerful figure on the Dark Side of the Force . '' Abrams had previously told Entertainment Weekly that the character , `` came to the name Kylo Ren when he joined a group called the Knights of Ren . '' Robbie Collin of The Telegraph described Ren as `` a hot - headed , radicalised Dark Side jihadi , whose red lightsaber splutters and crackles as violently as his temper '' . Abrams noted , `` The lightsaber is something that he built himself , and is as dangerous and as fierce and as ragged as the character . '' The Telegraph also explains that Ren 's wild and erratic temper and `` angsty '' instability make him dangerous . Melissa Leon of The Daily Beast describes Ren 's use of the Force as `` formidable '' , citing his ability to stop a blaster shot mid-air , immobilize victims and probe their minds against their will .
  Fact 1.0: Abrams told Empire in August 2015 that Kylo Ren is not a Sith.
  Fact 1.1: Abrams told Empire in August 2015 that Kylo Ren works under Supreme Leader Snoke.
  Fact 1.2: Abrams told Empire in August 2015 that Supreme Leader Snoke is a powerful figure on the Dark Side of the Force.
  Fact 1.3: Abrams had previously told Entertainment Weekly that the character came to the name Kylo Ren when he joined a group called the Knights of Ren.
  Fact 1.4: Robbie Collin of The Telegraph described Ren as a hot-headed, radicalised Dark Side jihadi.
  Fact 1.5: Robbie Collin of The Telegraph described Ren's red lightsaber as spluttering and crackling as violently as his temper.
  Fact 1.6: Abrams noted that the lightsaber is something Ren built himself.
  Fact 1.7: Abrams noted that the lightsaber is as dangerous, fierce and ragged as the character.
  Fact 1.8: The Telegraph explains that Ren's wild and erratic temper and angsty instability make him dangerous.
  Fact 1.9: Melissa Leon of The Daily Beast describes Ren's use of the Force as formidable.
  Fact 1.10: Melissa Leon cites Ren's ability to stop a blaster shot mid-air.
  Fact 1.11: Melissa Leon cites Ren's ability to immobilize victims.
  Fact 1.12: Melissa Leon cites Ren's ability to probe victims' minds against their will.

Retrieved items:
<item 1>
Kylo Ren, born Ben Solo, is Han and Leia’s son and Luke’s former Jedi pupil. Turned by Snoke, he joins the First Order and seeks to emulate Darth Vader, yet
remains conflicted and emotionally volatile. In *The Force Awakens*, he kills Han, fights Rey, and is separated from her as Starkiller Base collapses. In *The
Last Jedi*, he kills Snoke, becomes Supreme Leader, and attacks the Resistance; Luke’s Force projection enables their escape.
</item 1>

<item 2>
Kylo Ren, also known as Ben Solo, is a Star Wars character who first appeared in *The Force Awakens* (2015). He was created by Lawrence Kasdan and J.J. Abrams.
The film received coverage and awards recognition, and Kylo was voted the seventh-greatest movie villain in 2018. StarWars.com and Wookieepedia provide further
information.
</item 2>

<item 3>
Kylo Ren
You normally think of order , and structure , and full commitment and no hesitation ... he 's just a little bit more unpolished . It 's in his costume , in his
lightsaber -- how you kind of get the sense that it could just not work at any moment ; that it could just blow up . That 's kind of like a big metaphor for him
. Appearances ( edit ) Model of Kylo 's cross-bladed lightsaber on display at Star Wars Launch Bay at Disney 's Hollywood Studios ( 2015 ) . Feature films (
edit ) The Force Awakens ( 2015 ) ( edit ) Main article : Star Wars : The Force Awakens Kylo Ren first appears in The Force Awakens as the commander of the
First Order , a sinister regime that has risen from the remains of the old Galactic Empire . After arriving at Jakku to retrieve a map containing the
coordinates for the location of Luke Skywalker , Ren kills an old vicar known as Lor San Tekka and captures Resistance pilot Poe Dameron , who has also been
sent to recover the map by General Leia Organa .
</item 3>

<item 4>
Kylo Ren, portrayed by Adam Driver, is the chosen name of Ben Solo, son of Han Solo and Leia Organa and nephew of Luke Skywalker. Trained as a Jedi by Luke, he
turns to the dark side under Supreme Leader Snoke, seeking to emulate his grandfather, Darth Vader. Master of the Knights of Ren and First Order commander, he
later becomes its Supreme Leader. He is introduced in *The Force Awakens* (2015) and appears in *The Last Jedi*
</item 4>

<item 5>
Kylo Ren
Kylo Ren - wikipedia Kylo Ren Jump to : navigation , search `` Kylo '' redirects here . For the web browser , see Kylo ( web browser ) . Kylo Ren Ben Solo Star
Wars character Kylo Ren 's costume displayed on a mannequin First appearance The Force Awakens ( 2015 ) Created by Lawrence Kasdan J.J.
</item 5>

<item 6>
Kylo Ren’s arc recalls Jacen Solo, while his design resembles Revan. Abrams says Ren is not a Sith but serves the powerful Dark Side figure Supreme Leader
Snoke; he took his name after joining the Knights of Ren. Critics describe him as radicalized, hot-tempered, and unstable. His self-built red lightsaber is
ragged, crackling, and dangerous—reflecting his fierce, volatile nature.
</item 6>

<item 7>
Kylo Ren
Ren 's character arc shares similarities with that of the Star Wars expanded universe character Jacen Solo , the son of Han Solo and Princess Leia who threatens
the galaxy as a fallen Jedi . Additionally , critics have noted a resemblance between Ren 's character design and that of Revan , the protagonist of Knights of
the Old Republic . Character ( edit ) Abrams told Empire in August 2015 , `` Kylo Ren is not a Sith . He works under Supreme Leader Snoke , who is a powerful
figure on the Dark Side of the Force . '' Abrams had previously told Entertainment Weekly that the character , `` came to the name Kylo Ren when he joined a
group called the Knights of Ren . '' Robbie Collin of The Telegraph described Ren as `` a hot - headed , radicalised Dark Side jihadi , whose red lightsaber
splutters and crackles as violently as his temper '' . Abrams noted , `` The lightsaber is something that he built himself , and is as dangerous and as fierce
and as ragged as the character . '' The Telegraph also explains that Ren 's wild and erratic temper and `` angsty '' instability make him dangerous .
</item 7>

<item 8>
Kylo Ren
Though trained by his uncle Luke Skywalker as a Jedi , he has been seduced to the dark side of the Force by Supreme Leader Snoke and aspires to be as powerful
as his grandfather , Darth Vader . Kylo Ren is also the master of the Knights of Ren , as well as a commander and later the supreme leader of the First Order ,
an organization spawned from the fallen Galactic Empire . He is featured in The Force Awakens media and merchandising and appears in the film 's sequel , Star
Wars : The Last Jedi ( 2017 ) . Contents ( hide ) 1 Concept and creation 2 Character 3 Appearances 3.1 Feature films 3.1. 1 The Force Awakens ( 2015 ) 3.1. 2
The Last Jedi ( 2017 ) 3.2 Related works and promotional material 3.3 Novels 4 Reception 5 Family tree 6 References 7 External links Concept and creation ( edit
) After Lucasfilm was acquired by The Walt Disney Company in 2012 , the script for The Force Awakens was written by Lawrence Kasdan , J.J.
</item 8>

<item 9>
Kylo Ren
Abrams Michael Arndt Portrayed by Adam Driver Voiced by Adam Driver ( Disney Infinity 3.0 , Lego Star Wars : The Force Awakens , Lego Star Wars : The Resistance
Rises ) Matthew Wood ( Star Wars Battlefront II ) Roger Craig Smith ( Star Wars Battlefront II ) Information Full name Ben Solo Species Human Gender Male
Occupation Supreme Leader of the First Order Master of the Knights of Ren Former Jedi Padawan Affiliation First Order Knights of Ren New Jedi Order ( formerly )
Family Han Solo ( father ) Leia Organa ( mother ) Relatives Luke Skywalker ( maternal uncle ) Anakin Skywalker ( maternal grandfather ) Padmé Amidala ( maternal
grandmother ) Bail Organa ( adopted maternal grandfather ) Breha Organa ( adopted maternal grandmother ) Shmi Skywalker ( maternal great - grandmother )
Homeworld Chandrila Kylo Ren is a fictional character in the Star Wars franchise . Introduced in the 2015 film Star Wars : The Force Awakens , he is portrayed
by Adam Driver . Kylo Ren is the chosen name of Ben Solo , the son of original Star Wars trilogy characters Han Solo and Leia Organa .
</item 9>

<item 10>
Kylo Ren
Abrams and Michael Arndt , with no direct participation from Star Wars creator George Lucas . According to director Abrams , the First Order is inspired by
ODESSA , a theoretical network of SS officers who fled to Argentina following World War II . Driver 's casting in the film in an unnamed role was first
announced on April 29 , 2014 . Kylo Ren was first seen , but still not named , in the 88 - second The Force Awakens teaser trailer released by Lucasfilm on
November 28 , 2014 , wielding a jagged red lightsaber with a crossguard . The name Kylo Ren was revealed by Entertainment Weekly in a Lucasfilm - designed Topps
- style trading card mockup in December 2014 . A May 2015 Vanity Fair photo shoot by Annie Leibovitz confirmed that Driver would be portraying Kylo . In an
interview with Time , The Force Awakens costume designer Michael Kaplan said that Abrams requested that Ren 's mask be designed to be memorable for a child .
After numerous attempts to gain Abrams ' approval , the final design was selected .
</item 10>
=== END ITEM id=1413d5627b578f45 ===

=== ITEM id=9f874a2462013c5b ===
Gold sentences and facts:
Sentence 0: Costs are associated with particular goods using one of the several formulas , including specific identification , first - in first - out ( FIFO ) , or average cost . Costs include all costs of purchase , costs of conversion and other costs that are incurred in bringing the inventories to their present location and condition . Costs of goods made by the businesses include material , labor , and allocated overhead . The costs of those goods which are not yet sold are deferred as costs of inventory until the inventory is sold or written down in value .
  Fact 0.0: Costs are associated with particular goods using one of several formulas.
  Fact 0.1: The formulas include specific identification.
  Fact 0.2: The formulas include first-in first-out (FIFO).
  Fact 0.3: The formulas include average cost.
  Fact 0.4: Costs include all costs of purchase.
  Fact 0.5: Costs include all costs of conversion.
  Fact 0.6: Costs include other costs that are incurred in bringing the inventories to their present location and condition.
  Fact 0.7: Costs of goods made by businesses include material.
  Fact 0.8: Costs of goods made by businesses include labor.
  Fact 0.9: Costs of goods made by businesses include allocated overhead.
  Fact 0.10: Costs of goods which are not yet sold are deferred as costs of inventory.
  Fact 0.11: Costs of goods which are not yet sold are deferred until the inventory is sold.
  Fact 0.12: Costs of goods which are not yet sold are deferred until the inventory is written down in value.

Retrieved items:
<item 1>
Cost of goods sold
Determining costs requires keeping records of goods or materials purchased and any discounts on such purchase . In addition , if the goods are modified , the
business must determine the costs incurred in modifying the goods . Such modification costs include labor , supplies or additional material , supervision ,
quality control and use of equipment . Principles for determining costs may be easily stated , but application in practice is often difficult due to a variety
of considerations in the allocation of costs . Cost of goods sold may also reflect adjustments . Among the potential adjustments are decline in value of the
goods ( i.e. , lower market value than cost ) , obsolescence , damage , etc . When multiple goods are bought or made , it may be necessary to identify which
costs relate to which particular goods sold . This may be done using an identification convention , such as specific identification of the goods , first - in -
first - out ( FIFO ) , or average cost . Alternative systems may be used in some countries , such as last - in - first - out ( LIFO ) , gross profit method ,
retail method , or combinations of these .
</item 1>

<item 2>
Cost of goods sold
Cost of goods sold - wikipedia Cost of goods sold Jump to : navigation , search Accounting Historical cost Constant purchasing power Management Tax Major types
( show ) Auditing Cost Forensic Financial Fund Governmental Management Social Tax Key concepts ( show ) Accounting period Accrual Constant purchasing power
Economic entity Fair value Going concern Historical cost Matching principle Materiality Revenue recognition Unit of account Selected accounts ( show ) Assets
Cash Cost of goods sold Depreciation / Amortization Equity Expenses Goodwill Liabilities Profit Revenue Accounting standards ( show ) Generally - accepted
principles Generally - accepted auditing standards Convergence International Financial Reporting Standards International Standards on Auditing Management
Accounting Principles Financial statements ( show ) Annual report Balance sheet Cash - flow Equity Income Management discussion Notes to the financial
statements Bookkeeping ( show ) Bank reconciliation Debits and credits Double - entry system FIFO and LIFO Journal Ledger / General ledger T accounts Trial
balance Auditing ( show ) Financial Internal Firms Report People and organizations ( show ) Accountants Accounting organizations Luca Pacioli Development ( show
) History Research Positive accounting Sarbanes -- Oxley Act Business portal Cost of goods sold ( COGS ) refers to the carrying value of goods sold during a
particular period .
</item 2>

<item 3>
Cost of goods sold
Most businesses make more than one of a particular item . Thus , costs are incurred for multiple items rather than a particular item sold . Determining how much
of each of these components to allocate to particular goods requires either tracking the particular costs or making some allocations of costs . Parts and raw
materials are often tracked to particular sets ( e.g. , batches or production runs ) of goods , then allocated to each item . Labor costs include direct labor
and indirect labor . Direct labor costs are the wages paid to those employees who spend all their time working directly on the product being manufactured .
Indirect labor costs are the wages paid to other factory employees involved in production . Costs of payroll taxes and fringe benefits are generally included in
labor costs , but may be treated as overhead costs . Labor costs may be allocated to an item or set of items based on timekeeping records . Costs of materials
include direct raw materials , as well as supplies and indirect materials . Where non-incidental amounts of supplies are maintained , the taxpayer must keep
inventories of the supplies for income tax purposes , charging them to expense or cost of goods sold as used rather than as purchased .
</item 3>

<item 4>
Cost of goods sold
Both International and U.S. accounting standards require that certain abnormal costs , such as those associated with idle capacity , must be treated as expenses
rather than part of inventory . Discounts that must be deducted from the costs of purchased inventory are the following : Trade discounts ( reduction in the
price of goods that a manufacturer or wholesaler provides to a retailer ) - includes a discount that is always allowed , regardless of the time of payment .
Manufacturer 's rebates -- is based on the dealer 's purchases during the year . Cash discounts ( a reduction in the invoice price that the seller provides if
the dealer pays immediately or within a specified time ) -- it may reduce COGS , or it may be treated separately as gross income . Value added tax is generally
not treated as part of cost of goods sold if it may be used as an input credit or otherwise recoverable from the taxing authority . Cost of goods made by the
business ( edit ) The cost of goods produced in the business should include all costs of production . The key components of cost generally include : Parts , raw
materials and supplies used , Labor , including associated costs such as payroll taxes and benefits , and Overhead of the business allocable to production .
</item 4>

<item 5>
Cost of goods sold
Materials and labor may be allocated based on past experience , or standard costs . Where materials or labor costs for a period exceed the expected amount of
standard costs , a variance . Such variances are then allocated among cost of goods sold and remaining inventory at the end of the period . Determining overhead
costs often involves making assumptions about what costs should be associated with production activities and what costs should be associated with other
activities . Traditional cost accounting methods attempt to make these assumptions based on past experience and management judgment as to factual relationships
. Activity based costing attempts to allocate costs based on those factors that drive the business to incur the costs . Overhead costs are often allocated to
sets of produced goods based on the ratio of labor hours or costs or the ratio of materials used for producing the set of goods . Overhead costs may be referred
to as factory overhead or factory burden for those costs incurred at the plant level or overall burden for those costs incurred at the organization level .
Where labor hours are used , a burden rate or overhead cost per hour of labor may be added along with labor costs . Other methods may be used to associate
overhead costs with particular goods produced .
</item 5>

<item 6>
Cost of goods sold
Cost of goods sold may be the same or different for accounting and tax purposes , depending on the rules of the particular jurisdiction . Certain expenses are
included in COGS . Expenses that are included in COGS can not be deducted again as a business expense . COGS expenses include : The cost of products or raw
materials , including freight or shipping charges ; The cost of storing products the business sells ; Direct labor costs for workers who produce the products ;
and Factory overhead expenses . Depreciation Importance of inventories ( edit ) Inventories have a significant effect on profits . A business that makes or buys
goods to sell must keep track of inventories of goods under all accounting and income tax rules . An example illustrates why . Fred buys auto parts and resells
them . In 2008 , Fred buys $100 worth of parts . He sells parts for $80 that he bought for $30 , and has $70 worth of parts left . In 2009 , he sells the
remainder of the parts for $180 . If he keeps track of inventory , his profit in 2008 is $50 , and his profit in 2009 is $110 , or $160 in total .
</item 6>

<item 7>
Cost of goods sold
Her cost of goods sold depends on her inventory method . Under specific identification , the cost of goods sold is 10 + 12 , the particular costs of machines A
and C. If she uses FIFO , her costs are 20 ( 10 + 10 ) . If she uses average cost , her costs are 22 ( ( 10 + 10 + 12 + 12 ) / 4 x 2 ) . If she uses LIFO , her
costs are 24 ( 12 + 12 ) . Thus , her profit for accounting and tax purposes may be 20 , 18 , or 16 , depending on her inventory method . After the sales , her
inventory values are either 20 , 22 or 24 . After year end , Jane decides she can make more money by improving machines B and D. She buys and uses 10 of parts
and supplies , and it takes 6 hours at 2 per hour to make the improvements to each machine . Jane has overhead , including rent and electricity . She calculates
that the overhead adds 0.5 per hour to her costs .
</item 7>

<item 8>
Cost of goods sold
Costs are associated with particular goods using one of the several formulas , including specific identification , first - in first - out ( FIFO ) , or average
cost . Costs include all costs of purchase , costs of conversion and other costs that are incurred in bringing the inventories to their present location and
condition . Costs of goods made by the businesses include material , labor , and allocated overhead . The costs of those goods which are not yet sold are
deferred as costs of inventory until the inventory is sold or written down in value . Contents ( hide ) 1 Overview 2 Importance of inventories 3 Cost of goods
for resale 4 Cost of goods made by the business 5 Identification conventions 6 Example 7 Write - downs and allowances 8 Alternative views 9 Other terms 10 See
also 11 References 12 Further reading 12.1 Formal guidance Overview ( edit ) Many businesses sell goods that they have bought or produced . When the goods are
bought or produced , the costs associated with such goods are capitalized as part of inventory ( or stock ) of goods . These costs are treated as an expense in
the period the business recognizes income from sale of the goods .
</item 8>

<item 9>
Cost of goods sold
Other methods may be used to associate overhead costs with particular goods produced . Overhead rates may be standard rates , in which case there may be
variances , or may be adjusted for each set of goods produced . Identification conventions ( edit ) In some cases , the cost of goods sold may be identified
with the item sold . Ordinarily , however , the identity of goods is lost between the time of purchase or manufacture and the time of sale . Determining which
goods have been sold , and the cost of those goods , requires either identifying the goods or using a convention to assume which goods were sold . This may be
referred to as a cost flow assumption or inventory identification assumption or convention . The following methods are available in many jurisdictions for
associating costs with goods sold and goods still on hand : Specific identification . Under this method , particular items are identified , and costs are
tracked with respect to each item . This may require considerable recordkeeping . This method can not be used where the goods or items are indistinguishable or
fungible . Average cost . The average cost method relies on average unit cost to calculate cost of units sold and ending inventory .
</item 9>

<item 10>
Cost of goods sold
U.S. Internal Revenue Service Publication 334 , Tax Guide for Small Business , pages 27 -- 29 . U.S. Financial Accounting Standards Board ASC 330 . ( hide )
Accounting Type Financial accounting Cost accounting Statements Balance sheet Income statement Cash flow statement Terms Revenue Cost of goods sold Operating
expenses Capital expenditure Depreciation Gross profit Net profit Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Cost_of_goods_sold&oldid=834402233 '' Categories : Inventory Accounting terminology Costs Sales Hidden categories :
All articles with dead external links Articles with dead external links from April 2018 Webarchive template wayback links Talk Contents About Wikipedia Deutsch
فارسی Bahasa Indonesia עברית Lietuvių 日本 語 Polski Русский Українська 中文 3 more Edit links This page was last edited on 5 April 2018 , at 14 : 57 . Text is
available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and
Privacy Policy .
</item 10>
=== END ITEM id=9f874a2462013c5b ===
