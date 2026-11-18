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

This file has 10 items.

=== ITEM id=f9e5b5a5e5c8e5b0 ===
Gold sentences and facts:
Sentence 0: Minersville School District v. Gobitis, 310 U.S. 586 (1940) , was a decision by the Supreme Court of the United States involving the religious rights of public school students under the First Amendment to the United States Constitution.
  Fact 0.0: Minersville School District v. Gobitis, 310 U.S. 586 (1940), was a decision by the Supreme Court of the United States.
  Fact 0.1: Minersville School District v. Gobitis involved the religious rights of public school students.
  Fact 0.2: Minersville School District v. Gobitis involved religious rights under the First Amendment to the United States Constitution.
Sentence 1: Gravel v. United States, 408 U.S. 606 (1972), was a case regarding the protections offered by the Speech or Debate Clause of the United States Constitution.
  Fact 1.0: Gravel v. United States, 408 U.S. 606 (1972), was a case.
  Fact 1.1: Gravel v. United States concerned the protections offered by the Speech or Debate Clause of the United States Constitution.

Retrieved items:
<item 1>
Owasso Independent School District v. Falvo
Owasso Independent School District v. Falvo, 534 U.S. 426 (2002) , was a case in which the United States Supreme Court held in favor of the school district that
students scoring each other's tests and calling out the grades do not violate the Family Educational Rights and Privacy Act of 1974 (FERPA).

 Justice Kennedy wrote the opinion for the unanimous court.

 Justice Scalia wrote a concurring opinion in which he agreed with the ruling, but took issue with parts of Kennedy's opinion.
</item 1>

<item 2>
Minersville School District v. Gobitis
Minersville School District v. Gobitis, 310 U.S. 586 (1940) , was a decision by the Supreme Court of the United States involving the religious rights of public
school students under the First Amendment to the United States Constitution.

 The Court ruled that public schools could compel students—in this case, Jehovah's Witnesses—to salute the American Flag and recite the Pledge of Allegiance
despite the students' religious objections to these practices.

 This decision led to increased persecution of Witnesses in the United States.

 The Supreme Court overruled this decision a mere three years later, in "West Virginia State Board of Education v. Barnette", 319 U.S. 624 (1943).
</item 2>

<item 3>
Abington School District v. Schempp
Abington School District v. Schempp, 374 U.S. 203 (1963) , was a United States Supreme Court case in which the Court decided 8–1 in favor of the respondent,
Edward Schempp, and declared school-sponsored Bible reading in public schools in the United States to be unconstitutional.

 The Chief Justice of the Supreme Court during this case was Earl Warren.
</item 3>

<item 4>
McCollum v. Board of Education
McCollum v. Board of Education, 333 U.S. 203 (1948), was a landmark 1948 United States Supreme Court case related to the power of a state to use its
tax-supported public school system to aid religious instruction.

 The case was a test of the separation of church and state with respect to education.
</item 4>

<item 5>
Cedar Rapids Community School Dist. v. Garret F.
Cedar Rapids Community School District v. Garret F., 526 U.S. 66 (1999) , was a United States Supreme Court case in which the Court ruled that the related
services provision in the Individuals with Disabilities Education Act (IDEA) required public school districts to fund "continuous, one-on-one nursing care for
disabled children" despite arguments from the school district concerning the costs of the services."

 The judges relied heavily on Irving Independent School Dist.

 v. Tatro, 468 U. S. 883 (1984).

 Under the Court's reading of the IDEA's relevant provisions, medical treatments such as suctioning, ventilator checks, catheterization, and others which can be
administered by non-physician personnel come within the parameters of the special education law's related services.

 Disability advocates considered the Court decision to be a "substantial victory for families of children with disabilities."

 Amendments were made in the Education Flexibility Partnership Act of 1999 to increase IDEA funding as a result of the case.
</item 5>

<item 6>
Gravel v. United States
Gravel v. United States, 408 U.S. 606 (1972), was a case regarding the protections offered by the Speech or Debate Clause of the United States Constitution.

 In the case, the Supreme Court of the United States held that the privileges and immunities of the Constitution's Speech or Debate Clause enjoyed by members of
Congress also extend to Congressional aides, but not to activity outside the legislative process.
</item 6>

<item 7>
Elk Grove Unified School District v. Newdow
Elk Grove Unified School District v. Newdow, 542 U.S. 1 (2004), was a case decided by the U.S. Supreme Court.

 The lawsuit, originally filed as Newdow v. United States Congress, Elk Grove Unified School District, et al. in 2000, led to a 2002 ruling by the United States
Court of Appeals for the Ninth Circuit that the words "under God" in the Pledge of Allegiance are an endorsement of religion and therefore violate the
Establishment Clause of the First Amendment to the United States Constitution.

 After an initial decision striking the congressionally added "one nation under God" language, [Newdow v. U.S. Congress, 292 F.3d 597 (9th Cir.

 2002), the superseding opinion on denial of rehearing en banc was more limited, holding that compelled recitation of the language by school teachers to
students was invalid.

 [328 F.3d 466 (9th Cir.

 2003).
</item 7>

<item 8>
Safford Unified School District v. Redding
Safford Unified School District v. Redding, 557 U.S. 364 (2009) , was a United States Supreme Court case in which the Court held that a strip search of a middle
schooler violated the Fourth Amendment to the United States Constitution where the school lacked reasons to suspect either that the drugs (Ibuprofen) presented
a danger or that they were concealed in her underwear.

 The court also held, however, that because this was not clearly established law prior to the court's decision, the officials involved were shielded from
liability by qualified immunity.
</item 8>

<item 9>
Forest Grove School District v. T. A.
Forest Grove School District v. T. A., 557 U.S. 230 (2009), is a case in which the United States Supreme Court held that the Individuals with Disabilities
Education Act (IDEA) authorizes reimbursement for private special education services when a public school fails to provide a "free appropriate public education"
(FAPE) and the private school placement is appropriate, regardless of whether the child previously received special education services through the public
school.
</item 9>

<item 10>
Bethel School District v. Fraser
Bethel School District v. Fraser, 478 U.S. 675 (1986), was a landmark decision by the United States Supreme Court involving free speech in public schools.

 High school student Matthew Fraser was suspended from school in the Bethel School District for making a speech including sexual double entendres at a school
assembly.

 The Supreme Court held that his suspension did not violate the First Amendment.
</item 10>
=== END ITEM id=f9e5b5a5e5c8e5b0 ===

=== ITEM id=04e05f331e4ac6e5 ===
Gold sentences and facts:
Sentence 0: Dieffenbachia is a genus of tropical flowering plants in the family Araceae.
  Fact 0.0: Dieffenbachia is a genus of flowering plants.
  Fact 0.1: Dieffenbachia is a genus of tropical flowering plants.
  Fact 0.2: Dieffenbachia is in the family Araceae.
Sentence 1: Carlina is a genus of flowering plants in the aster family, Asteraceae.
  Fact 1.0: Carlina is a genus of flowering plants.
  Fact 1.1: Carlina is in the aster family.
  Fact 1.2: Carlina is in the family Asteraceae.

Retrieved items:
<item 1>
Xanthosoma
Xanthosoma is a genus of flowering plants in the arum family, Araceae.

 The genus is native to tropical America but widely cultivated and naturalized in other tropical regions.

 Several are grown for their starchy corms, an important food staple of tropical regions, known variously as "malanga", "otoy", "otoe", cocoyam (or new
cocoyam), "tannia", "tannier", "yautía", "macabo", "ocumo", "macal", "taioba", "dasheen", "quequisque", "ʻ ape" and (in Papua New Guinea) as Singapore taro
("taro kongkong").

 Many other species (including especially "X. roseum") are used as ornamental plants, and in popular horticultural literature are known as ‘ape or elephant ear
(from the purported resemblance of the leaf to an elephant's ear), although the latter name is sometimes also applied to members with similar appearance and
uses in the closely related genera "Caladium", "Colocasia" (i.e., taro), and "Alocasia".
</item 1>

<item 2>
Dictyosperma
Dictyosperma is a monotypic genus of flowering plant in the palm family found in the Mascarene Islands in the Indian Ocean (Mauritius, Réunion and Rodrigues).

 The sole species, Dictyosperma album, is widely cultivated in the tropics but has been farmed to near extinction in its native habitat.

 It is commonly called princess palm or hurricane palm, the latter owing to its ability to withstand strong winds by easily shedding leaves.

 It is closely related to, and resembles, palms in the "Archontophoenix" genus.

 The genus is named from two Greek words meaning "net" and "seed" and the epithet is Latin for "white", the common color of the crownshaft at the top of the
trunk.
</item 2>

<item 3>
Caladium
Caladium is a genus of flowering plants in the family Araceae.

 They are often known by the common name elephant ear (which they share with the closely related genera "Alocasia", "Colocasia", and "Xanthosoma"), Heart of
Jesus, and Angel Wings.

 There are over 1000 named cultivars of "Caladium bicolor" from the original South American plant.
</item 3>

<item 4>
Caryophyllales
Caryophyllales ( ) is an order of flowering plants that includes the cacti, carnations, amaranths, ice plants, beets, and many carnivorous plants.

 Many members are succulent, having fleshy stems or leaves.
</item 4>

<item 5>
Colocasia
Colocasia is a genus of flowering plants in the family Araceae, native to southeastern Asia and the Indian subcontinent.

 Some species are widely cultivated and naturalized in other tropical and subtropical regions.

 Common names include tarul, karkala ko ganu, elephant-ear, taro, cocoyam, dasheen, chembu, champadhumpa, shavige gadde, and eddoe.

 Elephant-ear and cocoyam are also used for some other large-leaved genera in the Araceae, notably "Xanthosoma" and "Caladium".

 The generic name is derived from the ancient Greek word "kolokasion", which in the Greek botanist Dioscorides (1st century AD) meant the edible roots of both
"Colocasia esculenta" and "Nelumbo nucifera".

 It is thought that the edible roots of "Colocasia esculenta" have been cultivated in Asia for more than ten thousand years.

 The species "Colocasia esculenta" is an invasive species in wetlands along the American Gulf coast, where it threatens to displace native wetland plants.
</item 5>

<item 6>
Dieffenbachia
Dieffenbachia is a genus of tropical flowering plants in the family Araceae.

 It is native to the New World Tropics from Mexico and the West Indies south to Argentina.

 Some species are widely cultivated as ornamental plants, especially as houseplants, and have become naturalized on a few tropical islands.
</item 6>

<item 7>
Bognera
Bognera is a monotypic genus of flowering plants in the Araceae family.

 The single species that makes up the genus is Bognera recondita.

 The word "recondita" means "hidden" referring to the fact that the plant is only found in remote areas of Amazonian Brazil near the Peruvian border.

 The species was discovered in the late 1970s and was originally placed in the genus "Ulearum".

 In 1984 a new genus, "Bognera" named after the German Aroid specialist Josef Bogner, was created for it.

 "Bognera" is believed to be closely related to "Dieffenbachia".
</item 7>

<item 8>
Cassytha
Cassytha is a genus of some two dozen species of obligately parasitic vines in the family Lauraceae.

 Superficially, and in some aspects of their ecology, they closely resemble plants in the unrelated genus "Cuscuta", the dodders.

 When fruit and flowers are absent in the field, the physical resemblance is so close that few people without technical training can tell them apart.

 In this respect and in their ecology the two genera present a spectacular example of convergent evolution.

 In its divergence from habits typical of the Lauraceae, "Cassytha" also presents examples of mosaic evolution
</item 8>

<item 9>
Vascular plant
Vascular plants (from Latin "vasculum": duct), also known as tracheophytes (from the equivalent Greek term "trachea") and also higher plants, form a large group
of plants (c. 308,312 accepted known species ) that are defined as those land plants that have lignified tissues (the xylem) for conducting water and minerals
throughout the plant.

 They also have a specialized non-lignified tissue (the phloem) to conduct products of photosynthesis.

 Vascular plants include the clubmosses, horsetails, ferns, gymnosperms (including conifers) and angiosperms (flowering plants).

 Scientific names for the group include Tracheophyta and Tracheobionta.
</item 9>

<item 10>
Carlina
Carlina is a genus of flowering plants in the aster family, Asteraceae.

 It is distributed from Madeira and the Canary Islands across Europe and northern Africa to Siberia and northwestern China.
</item 10>
=== END ITEM id=04e05f331e4ac6e5 ===

=== ITEM id=e81cea631d86e31b ===
Gold sentences and facts:
Sentence 0: Orlando Brown (born December 4, 1987) is an American actor, voice actor, rapper and singer.
  Fact 0.0: Orlando Brown was born on December 4, 1987.
  Fact 0.1: Orlando Brown is an American actor.
  Fact 0.2: Orlando Brown is a voice actor.
  Fact 0.3: Orlando Brown is a rapper.
  Fact 0.4: Orlando Brown is a singer.
Sentence 1:  He is best known for his role as Eddie Thomas in "That's So Raven", 3J Winslow in "Family Matters", Tiger in "Major Payne", Max in "Two of a Kind", Damey Wayne in the short-lived Waynehead, Dobbs in "Max Keeble's Big Move", and Frankie in "Eddie's Million Dollar Cook Off".
  Fact 1.0: Orlando Brown is best known for his role as Eddie Thomas in "That's So Raven".
  Fact 1.1: Orlando Brown is best known for his role as 3J Winslow in "Family Matters".
  Fact 1.2: Orlando Brown is best known for his role as Tiger in "Major Payne".
  Fact 1.3: Orlando Brown is best known for his role as Max in "Two of a Kind".
  Fact 1.4: Orlando Brown is best known for his role as Damey Wayne in the short-lived Waynehead.
  Fact 1.5: Orlando Brown is best known for his role as Dobbs in "Max Keeble's Big Move".
  Fact 1.6: Orlando Brown is best known for his role as Frankie in "Eddie's Million Dollar Cook Off".
Sentence 2: Family Matters is an American sitcom which originated on ABC from September 22, 1989 to May 9, 1997, before moving to CBS from September 19, 1997 to July 17, 1998.
  Fact 2.0: Family Matters is an American sitcom.
  Fact 2.1: Family Matters originated on ABC.
  Fact 2.2: Family Matters originated on ABC from September 22, 1989 to May 9, 1997.
  Fact 2.3: Family Matters moved to CBS after its ABC run.
  Fact 2.4: Family Matters was on CBS from September 19, 1997 to July 17, 1998.
Sentence 3:  A spin-off of "Perfect Strangers," the series revolves around the Winslow family, a middle-class African American family living in Chicago, Illinois.
  Fact 3.0: Family Matters is a spin-off of "Perfect Strangers."
  Fact 3.1: Family Matters revolves around the Winslow family.
  Fact 3.2: The Winslow family is a middle-class African American family.
  Fact 3.3: The Winslow family lives in Chicago, Illinois.

Retrieved items:
<item 1>
Darius McCrary
Darius Creston McCrary (born May 1, 1976) is an American film and television actor and singer.

 He is best known for his role as Edward "Eddie" Winslow, the oldest child of Carl and Harriette Winslow on the ABC/CBS television sitcom "Family Matters "which
ran from 1989–1998.

 Another one of McCrary's notable roles was as Scam in the 1987 comedy film "Big Shots", which was his film debut.

 He provided the voice of Jazz in the 2007 "Transformers" reboot.

 From December 2009 to October 2011, he portrayed photographer Malcolm Winters on the CBS daytime drama "The Young and the Restless".
</item 1>

<item 2>
Orlando Brown (actor)
Orlando Brown (born December 4, 1987) is an American actor, voice actor, rapper and singer.

 He is best known for his role as Eddie Thomas in "That's So Raven", 3J Winslow in "Family Matters", Tiger in "Major Payne", Max in "Two of a Kind", Damey Wayne
in the short-lived Waynehead, Dobbs in "Max Keeble's Big Move", and Frankie in "Eddie's Million Dollar Cook Off".
</item 2>

<item 3>
Eric Brown (actor)
Eric Brown (born December 17, 1964) is a former American film and television actor.

 Brown is best known for his role as Phillip Fillmore in the 1981 coming of age film "Private Lessons" and as Vinton "Buzz" Harper, Jr. in the NBC version of
the American sitcom "Mama's Family".
</item 3>

<item 4>
Alan Tudyk
Alan Wray Tudyk ( ; born March 16, 1971) is an American actor and voice actor known for his roles as Hoban "Wash" Washburne in the space western television
series "Firefly" and movie "Serenity", Alpha in the science fiction TV series "Dollhouse", Tucker McGee in "Tucker & Dale vs. Evil", Steve the Pirate in "", and
Wat in "A Knight's Tale".

 He also co-starred on the ABC sitcom "Suburgatory".

 Tudyk had a supporting role in the film "42" as baseball manager Ben Chapman and voiced roles in every Walt Disney Animation Studios film to date starting with
"Wreck-It Ralph".

 He has also featured in the web series "Con Man" and the films "" and "Rogue One: A Star Wars Story".
</item 4>

<item 5>
Jaimee Foxworth
Jaimee Monae Foxworth (born December 17, 1979) is an American actress and model.

 She is best known for her role of Judy Winslow, the youngest daughter of Carl and Harriette Winslow on the ABC/CBS sitcom "Family Matters".

 She later transitioned to pornographic films using the name Crave, starring in several titles.
</item 5>

<item 6>
Tyler Perry's For Better or Worse
Tyler Perry's For Better or Worse is an American television sitcom created, written and executive produced by Tyler Perry.

 The series is based on and is a TV spin-off to Perry's 2007 film "Why Did I Get Married?

" and its 2010 sequel "Why Did I Get Married Too?

".

 The show premiered on TBS on November 25, 2011 and ended on July 22, 2017.

 Led by the comical, over-the-top antics of Marcus and Angela Williams, the ensemble follows three couples: Marcus and Angela, Joseph and Leslie, and Richard
and Keisha who are at various stages of their relationships as they go through the ups-and-downs of married life and dating.

 The third season of the series premiered on September 18, 2013, on the Oprah Winfrey Network.
</item 6>

<item 7>
Randy J. Goodwin
Randy J. Goodwin, born as Ivan Jerome Goodwin, is a Hollywood actor.

 He is perhaps best known for his role as "Kennedy Winslow" on "Fast Track" and "Max Ellis" on the show "Abby", as well as his recurring roles on "Girlfriends"
as "Davis Hamilton" and "The Vampire Diaries" as "Dr. Jonas Martin".
</item 7>

<item 8>
Family Matters
Family Matters is an American sitcom which originated on ABC from September 22, 1989 to May 9, 1997, before moving to CBS from September 19, 1997 to July 17,
1998.

 A spin-off of "Perfect Strangers," the series revolves around the Winslow family, a middle-class African American family living in Chicago, Illinois.

 Midway through the first season, the show introduced the Winslows' nerdy neighbor Steve Urkel (Jaleel White), who quickly became its breakout character and
eventually the show's main character.

 Having run for nine seasons, "Family Matters" became the second longest-running non-animated US sitcom with a predominantly African American cast, behind only
"The Jeffersons" (11).

 Having aired 215 episodes, "Family Matters" is ranked third, behind only "Tyler Perry's House of Payne" (254), and "The Jeffersons" (253).
</item 8>

<item 9>
Michael Jai White
Michael Jai White (born November 10, 1967) is an American actor and martial artist who has appeared in numerous films and television series.

 He is the first African American to portray a major comic book superhero in a major motion picture, having starred as Al Simmons, the protagonist in the 1997
film "Spawn".

 White appeared as Marcus Williams in the Tyler Perry films "Why Did I Get Married?

" and "Why Did I Get Married Too?

", and starred as the character on the TBS/OWN comedy-drama television series "Tyler Perry's For Better or Worse".

 White portrayed Jax Briggs in "".

 White also portrayed boxer Mike Tyson in the 1995 HBO television movie "Tyson", and starred as Black Dynamite.
</item 9>

<item 10>
Lamman Rucker
Lamman Rucker (born October 6, 1971) is an American actor.

 Rucker began his career on the daytime soap operas "As the World Turns" and "All My Children", before roles in Tyler Perry's films "Why Did I Get Married?

", "Why Did I Get Married Too?

", and "Meet the Browns", and its television adaptation.

 In 2016, he began starring as Jacob Greenleaf in the Oprah Winfrey Network drama series, "Greenleaf".
</item 10>
=== END ITEM id=e81cea631d86e31b ===

=== ITEM id=2467572bcd4f1958 ===
Gold sentences and facts:
Sentence 0: Scott Richard Weiland ( ; born Scott Richard Kline; October 27, 1967 – December 3, 2015) was an American musician, singer and songwriter.
  Fact 0.0: Scott Richard Weiland was born Scott Richard Kline.
  Fact 0.1: Scott Richard Weiland was born on October 27, 1967.
  Fact 0.2: Scott Richard Weiland's date range ends on December 3, 2015.
  Fact 0.3: Scott Richard Weiland was an American musician.
  Fact 0.4: Scott Richard Weiland was an American singer.
  Fact 0.5: Scott Richard Weiland was an American songwriter.
Sentence 1: Richard Willis Hawley (born 17 January 1967) is an English guitarist, singer-songwriter and producer.
  Fact 1.0: Richard Willis Hawley was born on 17 January 1967.
  Fact 1.1: Richard Willis Hawley is English.
  Fact 1.2: Richard Willis Hawley is a guitarist.
  Fact 1.3: Richard Willis Hawley is a singer-songwriter.
  Fact 1.4: Richard Willis Hawley is a producer.

Retrieved items:
<item 1>
Graph query row:
person: Scott Weiland
occupation: rock singer
</item 1>

<item 2>
A Compilation of Scott Weiland Cover Songs
A Compilation of Scott Weiland Cover Songs is a compilation album and third solo studio album by Stone Temple Pilots frontman Scott Weiland, released on August
30, 2011.

 The album consists entirely of cover songs, featuring an array of covers of artists that inspired Weiland, such as David Bowie, The Beatles, The Rolling
Stones, and The Smiths.

 The album was originally set to be released together with Weiland's memoir "Not Dead & Not for Sale" but Weiland decided later to release the album by itself
exclusively in digital format.
</item 2>

<item 3>
Hugh Whitaker
Hugh Whitaker (born 18 May 1961) is an English musician and the former drummer for the British indie rock band The Housemartins.

 He replaced original drummer Chris Lang and drummed for the band's first album, "London 0 Hull 4", and its attendant single releases.

 He left the band before the recording of their second album, "The People Who Grinned Themselves to Death".

 Whitaker left the band on amicable terms and even participated in the promotional video for the band's first single without him, "Five Get Over Excited",
wherein he was kidnapped by his replacement, Dave Hemingway, and locked in a hessian sack.
</item 3>

<item 4>
&quot;Happy&quot; in Galoshes
"Happy" in Galoshes is the second solo album by American rock singer Scott Weiland.

 Weiland, known for his roles as the lead singer in Stone Temple Pilots and Velvet Revolver, released his first album, "12 Bar Blues" in 1998.

 Ten years later, ""Happy" in Galoshes" served as the official follow-up.

 Two versions have been released, a single disc and double disc deluxe version—the deluxe edition offers a second disc of 10 extra tracks.

 The album was released November 25, 2008 on Weiland's own Softdrive Records.

 Produced by Doug Grean and Scott Weiland, with select tracks recorded by Steve Albini, the album features guest appearances by Paul Oakenfold and No Doubt
members Adrian Young, Tony Kanal, and Tom Dumont.

 Oakenfold appears on a cover of David Bowie's "Fame."

 Weiland has cited Bowie as one of his main influences.
</item 4>

<item 5>
Scott Weiland and the Wildabouts
Scott Weiland and the Wildabouts (sometimes billed simply as The Wildabouts) were a rock and roll band formed sometime around 2012 by former Stone Temple Pilots
and Velvet Revolver lead singer Scott Weiland, along with bassist Tommy Baker, guitarist Jeremy Brown, and multi-instrumentalist Doug Grean.

 The Wildabouts were marred with several roster changes (including the death of guitarist Jeremy Brown) and ultimately disbanded after Weiland's sudden death on
a tour bus in late 2015.
</item 5>

<item 6>
Scott Weiland
Scott Richard Weiland ( ; born Scott Richard Kline; October 27, 1967 – December 3, 2015) was an American musician, singer and songwriter.

 During a career spanning three decades, Weiland was best known as the lead singer of the band Stone Temple Pilots from 1989 to 2002 and 2008 to 2013.

 He was also a member of supergroup Velvet Revolver from 2003 to 2008 and recorded one album with another supergroup, Art of Anarchy.

 He established himself as a solo artist as well, releasing three studio albums, two cover albums, and collaborations with several other musicians throughout
his career.
</item 6>

<item 7>
Graph query row:
person: Richard Hawley
occupation: ['guitarist', 'singer-songwriter', 'producer', 'solo musician']
</item 7>

<item 8>
Doug Grean
Doug Grean is an American record producer, audio engineer, guitarist, based in Los Angeles.

 He has worked with artists Sheryl Crow, Ricki Lee Jones, Glen Campbell, Scott Weiland, Slash, Velvet Revolver, Stone Temple Pilots, The Crystal Method, Tim
McGraw, Cyndi Lauper, Annabella Lwin (Bow Wow Wow), Corey Glover (Living Color), Camp Freddy, DJ Hurricane and John Taylor (Duran Duran).

 Grean was formerly the lead guitarist and touring musical director for Scott Weiland and the Wildabouts.

 Some of Grean's television appearances with Weiland include: VH1 Legends, Last Call with Carson Daly, The Tonight Show with Jay Leno, and the Grammy awards
with Velvet Revolver as keyboardist, performing "All Across the Universe" with Stevie Wonder, Norah Jones, Bono, Steven Tyler, Brian Wilson, and Alison Krauss.
</item 8>

<item 9>
Richard Hawley
Richard Willis Hawley (born 17 January 1967) is an English guitarist, singer-songwriter and producer.

 After his first band Treebound Story (formed while he was still at school) broke up, Hawley found success as a member of Britpop band Longpigs in the 1990s.

 After that group broke up in 2000, he later joined the band Pulp, led by his friend Jarvis Cocker, for a short time.

 As a solo musician, Hawley has released seven studio albums.

 He has been nominated for a Mercury prize twice and once for a Brit Award.

 He has collaborated with Lisa Marie Presley, Arctic Monkeys, Manic Street Preachers and Paul Weller.
</item 9>

<item 10>
Bill Scott (author)
William Neville "Bill" Scott OAM (4 October 1923 – 22 December 2005) was an Australian author, folklorist, songwriter, poet and a collector of bush ballads and
Australian folk history.

 He has published anthologies of Australian bush songs, including the best selling book "The Complete Book of Australian Folklore" published in 1976.

 He was awarded the Order of Australia in 1992 for his contributions to folklore, folk music and Australian literature.

 He was considered a living treasure and his anthologies of songs and his donated collections continue his legacy.
</item 10>
=== END ITEM id=2467572bcd4f1958 ===

=== ITEM id=13f6ef6e409d4a92 ===
Gold sentences and facts:
Sentence 0: "Into Dust" is a song by American duo Mazzy Star that appears as the ninth track on their second album "So Tonight That I Might See".
  Fact 0.0: 'Into Dust' is a song by the American duo Mazzy Star.
  Fact 0.1: 'Into Dust' appears as the ninth track on the second album of Mazzy Star.
  Fact 0.2: The second album of Mazzy Star is 'So Tonight That I Might See'.
Sentence 1: Mazzy Star is an American alternative rock band formed in Santa Monica, California, in 1989 from the group Opal, a collaboration of guitarist David Roback and bassist Kendra Smith.
  Fact 1.0: Mazzy Star is an American alternative rock band.
  Fact 1.1: Mazzy Star was formed in Santa Monica, California, in 1989.
  Fact 1.2: Mazzy Star was formed from the group Opal.
  Fact 1.3: Opal was a collaboration of guitarist David Roback and bassist Kendra Smith.

Retrieved items:
<item 1>
Assembly of Dust
Assembly of Dust (also known as AOD) is an American rock band formed in 2002 by former Strangefolk frontman Reid Genauer.

 After the breakup of Strangefolk, Genauer decided to record a solo album, and he recruited some friends from Strangefolk's days on the road to help out.

 After titling the 2003 solo release “Assembly of Dust,” Genauer decided to use the name for his new group.

 AOD originally featured Genauer on lead vocals and guitar, Nate Wilson on the keyboard, Adam Terrell on lead guitar, John Leccese on bass, and Andy Herrick on
drums.

 In recent years the band has experienced some transformations, and the current status of the band now features Reid Genauer on lead vocals and guitar, Adam
Terrell on lead guitar, John Leccese on bass, Jason Crosby on the keys and violin, and Dave Diamond on the drums.

 Despite the changes in the band's lineup, AOD has managed to maintain the same high energy and tight musical finesse for which they are known.
</item 1>

<item 2>
Rock Star (TV series)
Rock Star is a television series produced by Mark Burnett, David Goffin, and Lisa Hennessy in which aspiring singers from all around the world competed to
become the lead singer of a featured group.

 It debuted on CBS on July 11, 2005, to mediocre ratings.

 The show was hosted by television personality and commercial spokeswoman Brooke Burke and Jane's Addiction & Red Hot Chili Peppers lead guitarist Dave Navarro.

 In Australian band INXS chose J.D. Fortune as their new lead singer.

 For , the band Rock Star Supernova chose Lukas Rossi as the lead singer of their new supergroup.
</item 2>

<item 3>
Cosmic Dust (band)
Cosmic Dust, also known as the Cosmic Dust Fusion Band, is an instrumental jazz band formed in 1990 by Jim Templeton.

 The band was the first well-known group that guitarist Myles Kennedy played in.

 The original lineup consisted of Jim Templeton on keyboard, Gary Edighoffer on saxophone, Clipper Anderson on double bass, Myles Kennedy on guitar, and Scott
Reusser on drums.

 Kennedy eventually left the band and went on to become the lead vocalist/lead guitarist for a jazz fusion group called Citizen Swing and later an alternative
rock band called The Mayfield Four.

 Kennedy is now fronting and playing guitar for the hard rock/alternative metal band Alter Bridge, which he helped form with Creed members Mark Tremonti, Scott
Phillips, and Brian Marshall in 2004, and is also the lead vocalist for Slash's solo band on tour.
</item 3>

<item 4>
Mazzy Star
Mazzy Star is an American alternative rock band formed in Santa Monica, California, in 1989 from the group Opal, a collaboration of guitarist David Roback and
bassist Kendra Smith.

 Roback's friend Hope Sandoval became the group's vocalist when Smith left the band.
</item 4>

<item 5>
Blitz Kids
The Blitz Kids were a group of young people who frequented the weekly Blitz club-night in Covent Garden, London in 1979-80, and are credited with launching the
New Romantic subcultural movement.

 It was Andrew Czezowski (Ex manager of The Damned) and Susan Carrington who went on to start The Roxy club with Barry Jones that introduced Steve Strange and
Rusty Egan to the Blitz in Covent Garden, which started the New Romantic Movement.

 Among the many attendees were Rusty Egan, Steve Strange, Boy George, Marilyn, Alice Temple, Perri Lister, Princess Julia, Philip Sallon, Carl Teper and Martin
Degville (later to be the frontman of Tony James' Sigue Sigue Sputnik).

 Crucially, the club lay between two art colleges (St Martin's School and Central School) and became a testbed for student fashion designers who set London
ablaze during the 1980s.
</item 5>

<item 6>
Into Dust
"Into Dust" is a song by American duo Mazzy Star that appears as the ninth track on their second album "So Tonight That I Might See".

 The song's arrangement is sparse and features only acoustic guitar and violin as accompaniment to the vocals.
</item 6>

<item 7>
Man or Astro-man?
Man or Astro-man?

 is a surf rock group that formed in Auburn, Alabama, in the early 1990s and came to prominence over the following decade.
</item 7>

<item 8>
Distortion Mirrors
Distortion Mirrors is an English alternative rock band, based in Leeds, West Yorkshire, England, and who was formed in 2011.
</item 8>

<item 9>
I Was Totally Destroying It
I Was Totally Destroying It is a five-piece powerpop band from Chapel Hill, North Carolina, featuring current and former members of Saddle Creek band Sorry
About Dresden.

 After self-releasing and touring the country behind their first full-length album, the band signed with Portland-based indie label Greyday Records.

 Since forming in 2007, the band has toured much of the country, playing with artists such as Conor Oberst and the Mystic Valley Band, Motion City Soundtrack,
Cursive, Superchunk, Portastatic, Ryan Ferguson (formerly of No Knife), Dear and the Headlights, Margot & the Nuclear So and So's, Snowden, Billy Bragg,
Annuals, Joan Jett, The Actual, Ozma, Polvo, Birds of Avalon and Blankface.

 Their next full-length, Horror Vacui, was released by Greyday in October 2009, and was co-produced by Josh Cain of Motion City Soundtrack.
</item 9>

<item 10>
American Hi-Fi
American Hi-Fi is an American rock band formed in Boston in 1998.

 The band consists of lead vocalist/rhythm guitarist Stacy Jones, lead guitarist Jamie Arentzen, bassist/backing vocalist Drew Parsons, and drummer Brian Nolan.

 Prior to the group's formation, Stacy Jones was well known for being a drummer in the successful alternative rock bands Veruca Salt and Letters to Cleo.

 American Hi-Fi has a close relationship with Miley Cyrus, whose band shares two members with American Hi-Fi.

 The group has a mixed musical style that includes influences from pop punk, alternative rock, and power pop.
</item 10>
=== END ITEM id=13f6ef6e409d4a92 ===

=== ITEM id=79326d7309e6f8e6 ===
Gold sentences and facts:
Sentence 0:  He has acted in a few Off Broadway productions and has made appearances on American television shows such as "", "Person of Interest" and "" and video games such as "Grand Theft Auto V", "Red Dead Redemption", "" and "Red Dead Revolver".
  Fact 0.0: This person has acted in a few Off Broadway productions.
  Fact 0.1: This person has made appearances on American television shows such as Person of Interest.
  Fact 0.2: This person has made appearances on an American television show whose title is blank in the sentence.
  Fact 0.3: This person has made appearances in video games such as Grand Theft Auto V.
  Fact 0.4: This person has made appearances in video games such as Red Dead Redemption.
  Fact 0.5: This person has made appearances in video games such as Red Dead Revolver.
  Fact 0.6: This person has made appearances in a video game whose title is blank in the sentence.
Sentence 1: Red Dead Redemption is a Western action-adventure video game developed by Rockstar San Diego and published by Rockstar Games.
  Fact 1.0: Red Dead Redemption is a Western action-adventure video game.
  Fact 1.1: Red Dead Redemption was developed by Rockstar San Diego.
  Fact 1.2: Red Dead Redemption was published by Rockstar Games.

Retrieved items:
<item 1>
Development of Grand Theft Auto V
An approximately 1,000-person team developed "Grand Theft Auto V", an action-adventure video game, over several years.

 Rockstar Games released "Grand Theft Auto V" on 17 September 2013 for PlayStation 3 and Xbox 360, on 18 November 2014 for PlayStation 4 and Xbox One, and on 14
April 2015 for Microsoft Windows, as the fifteenth entry in the "Grand Theft Auto" series.

 Development began soon after "Grand Theft Auto IV"' s release and was led by Rockstar North's core 360-person team, who collaborated with several other
international Rockstar Games studios.

 The team considered the game a spiritual successor to many of their previous projects like "Red Dead Redemption" and "Max Payne 3".

 After its unexpected announcement in 2011, the game was fervently promoted with press showings, cinematic trailers, viral marketing strategies and special
editions. Its release date, though subject to several delays, was widely anticipated.
</item 1>

<item 2>
Red Dead
Red Dead is a series of western action-adventure video games developed by Rockstar San Diego and published by Rockstar Games.

 The first game, titled "Red Dead Revolver", was originally released for the PlayStation 2 and Xbox in May 2004.

 The second game, "Red Dead Redemption", was released for the PlayStation 3 and Xbox 360 in May 2010 to wide critical acclaim, and is considered one of the
greatest games of all time.

 The third entry in the series, "Red Dead Redemption II", is scheduled for release in Q2 2018 for PlayStation 4 and Xbox One.
</item 2>

<item 3>
List of Grand Theft Auto V characters
Rockstar North's 2013 open world action-adventure video game "Grand Theft Auto V", published by Rockstar Games, deals with a trio of criminals and their efforts
to commit heists while under pressure from a corrupt government agency.

 The game's use of three lead protagonists is a break from series tradition.

 Three days after its release, "Grand Theft Auto V" had earned more than US $1 billion in sales, making it the fastest selling video game in history.

 Additionally, "Grand Theft Auto V" was voted Game of the Year from several gaming publications.

 The game's characters were created by writers Dan Houser and Rupert Humphries.

 The use of multiple lead protagonists, as well as the cast in general, has received generally positive reviews from gaming magazines and websites.
</item 3>

<item 4>
Rockstar’s Red Dead series began with 2004’s Revolver, continued with 2010’s acclaimed Redemption, and was set to add 2018 prequel Redemption II, starring
Arthur Morgan. Grand Theft Auto V, developed by roughly 1,000 people, launched across consoles and PC from 2013–15; its three-protagonist heist story earned
acclaim, record-breaking sales, and awards, but drew criticism over violence, torture, women’s portrayal, and alleged
</item 4>

<item 5>
Controversies surrounding Grand Theft Auto V
"Grand Theft Auto V" is an open world, action-adventure video game developed by Rockstar North and published by Rockstar Games.

 Upon its 17 September 2013 release on PlayStation 3 and Xbox 360, the game generated controversies related to its violence and depiction of women.

 A mission that requires players to use torture equipment in a hostage interrogation polarised reviewers, who noted its political commentary but felt that the
torture sequence was in poor taste.

 The mission also received criticism from politicians and anti-torture charity groups.

 The game became subject to widespread online debate over its portrayal of women, particularly in the wake of backlash against GameSpot journalist Carolyn Petit
after she claimed the game was misogynystic in her review.

 After Petit's review webpage received more than 20,000 largely negative comments, many journalists defended her right to an opinion and lamented the gaming
community's defensiveness towards criticism.

 Television personality Karen Gravano and actress Lindsay Lohan both filed lawsuits against Rockstar in allegation that characters in the game were based on
their likenesses.
</item 5>

<item 6>
Red Dead Redemption II
Red Dead Redemption II is an upcoming western action-adventure video game developed and published by Rockstar Games for release on PlayStation 4 and Xbox One in
Q2 2018.

 The game is a prequel to the 2010 title "Red Dead Redemption" and will be the third entry in the "Red Dead" series.

 It follows outlaw Arthur Morgan, a member of the Dutch van der Linde gang.
</item 6>

<item 7>
Red Dead Revolver
Red Dead Revolver is a 2004 western action-adventure third-person shooter video game developed by Rockstar San Diego, published by Rockstar Games and
distributed by Take-Two Interactive for the PlayStation 2 and Xbox.

 It is the first game in the "Red Dead" series.
</item 7>

<item 8>
Red Dead Redemption
Red Dead Redemption is a Western action-adventure video game developed by Rockstar San Diego and published by Rockstar Games.

 It was released for PlayStation 3 and Xbox 360 consoles in May 2010.

 It is the second title in the "Red Dead" franchise, after 2004's "Red Dead Revolver".

 The game, set during the decline of the American Frontier in the year 1911, follows John Marston, a former outlaw whose wife and son are taken hostage by the
government in ransom for his services as a hired gun.

 Having no other choice, Marston sets out to bring the three members of his former gang to justice.
</item 8>

<item 9>
List of accolades received by Grand Theft Auto V
"Grand Theft Auto V" is an open world, action-adventure video game developed by Rockstar North and published by Rockstar Games.

 Players freely roam the fictional state of San Andreas (based on Southern California), composed of open countryside and the fictional city of Los Santos (based
on Los Angeles).

 The game was first announced on 25 October 2011, and was widely anticipated.

 At the 2012 Spike Video Game Awards (now the VGX), it was awarded Most Anticipated Game.

 It was released on 17 September 2013 for the PlayStation 3 and Xbox 360, on 18 November 2014 for the PlayStation 4 and Xbox One, and on 14 April 2015 for
Microsoft Windows.

 Review aggregator, Metacritic, assigned the game a normalised score of 97 out of 100, indicating "universal acclaim", based on 50 reviews for the PlayStation 3
version, 58 reviews for the Xbox 360 version, 66 reviews for the PlayStation 4 version and 14 reviews for the Xbox One version.
</item 9>

<item 10>
Tyler Bunch
Tyler T. Bunch (born February 10, 1970) is an American puppeteer, puppet designer, director, and actor.

 In addition to his work for the Jim Henson Company, Bunch has created and performed puppets for PBS, Nickelodeon and Disney's The Muppets Studio.

 He has acted in a few Off Broadway productions and has made appearances on American television shows such as "", "Person of Interest" and "" and video games
such as "Grand Theft Auto V", "Red Dead Redemption", "" and "Red Dead Revolver".

 Currently, he is directing the PBS children's television series "SeeMore's Playhouse".

 He is also currently scheduled to be one of the puppeteers for the musical stage adaptation of "Emmet Otter's Jug-Band Christmas".

 He resides in Stratford, Connecticut.
</item 10>
=== END ITEM id=79326d7309e6f8e6 ===

=== ITEM id=ff41f00279270b67 ===
Gold sentences and facts:
Sentence 0:  He has acted in a few Off Broadway productions and has made appearances on American television shows such as "", "Person of Interest" and "" and video games such as "Grand Theft Auto V", "Red Dead Redemption", "" and "Red Dead Revolver".
  Fact 0.0: This person has acted in a few Off Broadway productions.
  Fact 0.1: This person has made appearances on American television shows such as Person of Interest.
  Fact 0.2: This person has made appearances on an American television show whose title is blank in the sentence.
  Fact 0.3: This person has made appearances in video games such as Grand Theft Auto V.
  Fact 0.4: This person has made appearances in video games such as Red Dead Redemption.
  Fact 0.5: This person has made appearances in video games such as Red Dead Revolver.
  Fact 0.6: This person has made appearances in a video game whose title is blank in the sentence.
Sentence 1: Red Dead Redemption is a Western action-adventure video game developed by Rockstar San Diego and published by Rockstar Games.
  Fact 1.0: Red Dead Redemption is a Western action-adventure video game.
  Fact 1.1: Red Dead Redemption was developed by Rockstar San Diego.
  Fact 1.2: Red Dead Redemption was published by Rockstar Games.

Retrieved items:
<item 1>
Red Dead Redemption II
Red Dead Redemption II is an upcoming western action-adventure video game developed and published by Rockstar Games for release on PlayStation 4 and Xbox One in
Q2 2018.

 The game is a prequel to the 2010 title "Red Dead Redemption" and will be the third entry in the "Red Dead" series.

 It follows outlaw Arthur Morgan, a member of the Dutch van der Linde gang.
</item 1>

<item 2>
Red Dead
Red Dead is a series of western action-adventure video games developed by Rockstar San Diego and published by Rockstar Games.

 The first game, titled "Red Dead Revolver", was originally released for the PlayStation 2 and Xbox in May 2004.

 The second game, "Red Dead Redemption", was released for the PlayStation 3 and Xbox 360 in May 2010 to wide critical acclaim, and is considered one of the
greatest games of all time.

 The third entry in the series, "Red Dead Redemption II", is scheduled for release in Q2 2018 for PlayStation 4 and Xbox One.
</item 2>

<item 3>
Red Dead Redemption
Red Dead Redemption is a Western action-adventure video game developed by Rockstar San Diego and published by Rockstar Games.

 It was released for PlayStation 3 and Xbox 360 consoles in May 2010.

 It is the second title in the "Red Dead" franchise, after 2004's "Red Dead Revolver".

 The game, set during the decline of the American Frontier in the year 1911, follows John Marston, a former outlaw whose wife and son are taken hostage by the
government in ransom for his services as a hired gun.

 Having no other choice, Marston sets out to bring the three members of his former gang to justice.
</item 3>

<item 4>
Grand Theft Auto: Vice City Stories
Grand Theft Auto: Vice City Stories is an open world action-adventure video game developed by Rockstar Leeds in conjunction with Rockstar North and published by
Rockstar Games.

 Released on 31 October 2006 for the PlayStation Portable and later for the PlayStation 2 on 5 March 2007.

 The game is the tenth instalment in the "Grand Theft Auto" series and the sixth and last game in the 3D universe.

 This was also the last game in the series to use well-known and Hollywood voice talent; "Grand Theft Auto IV" onward would instead use lesser-known and unknown
voice actors.

 It is a prequel to "" and successor to the previous PSP release, "".

 Vice City is a representation of Miami, Florida.
</item 4>

<item 5>
Controversies surrounding Grand Theft Auto V
"Grand Theft Auto V" is an open world, action-adventure video game developed by Rockstar North and published by Rockstar Games.

 Upon its 17 September 2013 release on PlayStation 3 and Xbox 360, the game generated controversies related to its violence and depiction of women.

 A mission that requires players to use torture equipment in a hostage interrogation polarised reviewers, who noted its political commentary but felt that the
torture sequence was in poor taste.

 The mission also received criticism from politicians and anti-torture charity groups.

 The game became subject to widespread online debate over its portrayal of women, particularly in the wake of backlash against GameSpot journalist Carolyn Petit
after she claimed the game was misogynystic in her review.

 After Petit's review webpage received more than 20,000 largely negative comments, many journalists defended her right to an opinion and lamented the gaming
community's defensiveness towards criticism.

 Television personality Karen Gravano and actress Lindsay Lohan both filed lawsuits against Rockstar in allegation that characters in the game were based on
their likenesses.
</item 5>

<item 6>
Tyler Bunch
Tyler T. Bunch (born February 10, 1970) is an American puppeteer, puppet designer, director, and actor.

 In addition to his work for the Jim Henson Company, Bunch has created and performed puppets for PBS, Nickelodeon and Disney's The Muppets Studio.

 He has acted in a few Off Broadway productions and has made appearances on American television shows such as "", "Person of Interest" and "" and video games
such as "Grand Theft Auto V", "Red Dead Redemption", "" and "Red Dead Revolver".

 Currently, he is directing the PBS children's television series "SeeMore's Playhouse".

 He is also currently scheduled to be one of the puppeteers for the musical stage adaptation of "Emmet Otter's Jug-Band Christmas".

 He resides in Stratford, Connecticut.
</item 6>

<item 7>
Development of Grand Theft Auto V
An approximately 1,000-person team developed "Grand Theft Auto V", an action-adventure video game, over several years.

 Rockstar Games released "Grand Theft Auto V" on 17 September 2013 for PlayStation 3 and Xbox 360, on 18 November 2014 for PlayStation 4 and Xbox One, and on 14
April 2015 for Microsoft Windows, as the fifteenth entry in the "Grand Theft Auto" series.

 Development began soon after "Grand Theft Auto IV"' s release and was led by Rockstar North's core 360-person team, who collaborated with several other
international Rockstar Games studios.

 The team considered the game a spiritual successor to many of their previous projects like "Red Dead Redemption" and "Max Payne 3".

 After its unexpected announcement in 2011, the game was fervently promoted with press showings, cinematic trailers, viral marketing strategies and special
editions. Its release date, though subject to several delays, was widely anticipated.
</item 7>

<item 8>
List of accolades received by Grand Theft Auto V
"Grand Theft Auto V" is an open world, action-adventure video game developed by Rockstar North and published by Rockstar Games.

 Players freely roam the fictional state of San Andreas (based on Southern California), composed of open countryside and the fictional city of Los Santos (based
on Los Angeles).

 The game was first announced on 25 October 2011, and was widely anticipated.

 At the 2012 Spike Video Game Awards (now the VGX), it was awarded Most Anticipated Game.

 It was released on 17 September 2013 for the PlayStation 3 and Xbox 360, on 18 November 2014 for the PlayStation 4 and Xbox One, and on 14 April 2015 for
Microsoft Windows.

 Review aggregator, Metacritic, assigned the game a normalised score of 97 out of 100, indicating "universal acclaim", based on 50 reviews for the PlayStation 3
version, 58 reviews for the Xbox 360 version, 66 reviews for the PlayStation 4 version and 14 reviews for the Xbox One version.
</item 8>

<item 9>
Red Dead Revolver
Red Dead Revolver is a 2004 western action-adventure third-person shooter video game developed by Rockstar San Diego, published by Rockstar Games and
distributed by Take-Two Interactive for the PlayStation 2 and Xbox.

 It is the first game in the "Red Dead" series.
</item 9>

<item 10>
List of Grand Theft Auto V characters
Rockstar North's 2013 open world action-adventure video game "Grand Theft Auto V", published by Rockstar Games, deals with a trio of criminals and their efforts
to commit heists while under pressure from a corrupt government agency.

 The game's use of three lead protagonists is a break from series tradition.

 Three days after its release, "Grand Theft Auto V" had earned more than US $1 billion in sales, making it the fastest selling video game in history.

 Additionally, "Grand Theft Auto V" was voted Game of the Year from several gaming publications.

 The game's characters were created by writers Dan Houser and Rupert Humphries.

 The use of multiple lead protagonists, as well as the cast in general, has received generally positive reviews from gaming magazines and websites.
</item 10>
=== END ITEM id=ff41f00279270b67 ===

=== ITEM id=c9ed24899457530c ===
Gold sentences and facts:
Sentence 0: Batman is a 1989 American superhero film directed by Tim Burton and produced by Jon Peters and Peter Guber, based on the DC Comics character of the same name.
  Fact 0.0: Batman is a 1989 American superhero film.
  Fact 0.1: Batman was directed by Tim Burton.
  Fact 0.2: Batman was produced by Jon Peters and Peter Guber.
  Fact 0.3: Batman is based on the DC Comics character of the same name.
Sentence 1:  He is Chairman and CEO of Mandalay Entertainment.
  Fact 1.0: The person referred to as 'he' is Chairman of Mandalay Entertainment.
  Fact 1.1: The person referred to as 'he' is CEO of Mandalay Entertainment.

Retrieved items:
<item 1>
Batman: Return of the Joker
Batman: Return of the Joker, known in Japan as Dynamite Batman (ダイナマイトバットマン , Dainamaito Battoman ) , is a 1991 platform video game, the follow-up to Sunsoft's
on the Nintendo Entertainment System.

 Unlike that game, which was based on the 1989 Batman film directed by Tim Burton, "Return of the Joker" is entirely self-contained and based more on the modern
comic book iteration of Batman.

 However, Batman mans the Batmobile from the 1989 film.

 A remake of "Return of the Joker", titled Batman: Revenge of the Joker, was released on the Sega Genesis by Ringler Studios in 1992.

 A Super NES version of "Revenge of the Joker" was in development, but never released.
</item 1>

<item 2>
Batman (1989 film)
Batman is a 1989 American superhero film directed by Tim Burton and produced by Jon Peters and Peter Guber, based on the DC Comics character of the same name.

 It is the first installment of Warner Bros.' initial "Batman" film series.

 The film stars Michael Keaton as Bruce Wayne/Batman and Jack Nicholson as The Joker, alongside Kim Basinger, Robert Wuhl, Pat Hingle, Billy Dee Williams,
Michael Gough and Jack Palance.

 The film takes place early in the title character's war on crime, and depicts a battle with his arch-nemesis the Joker.
</item 2>

<item 3>
Batman (album)
Batman is the eleventh studio album by American recording artist Prince and the soundtrack album to the 1989 film "Batman".

 It was released on June 20, 1989 by Warner Bros.

 Records.

 As a Warner Bros. stablemate, Prince's involvement in the soundtrack was designed to leverage the media company's contract-bound talent as well as fulfill the
artist's need for a commercial (if not critical) revival.

 The result was yet another multi-platinum successful cross-media enterprise by Warner Bros., in the vein of "Purple Rain".

 The album was No. 1 on the "Billboard" albums chart for six consecutive weeks.

 It has sold over eleven million copies worldwide.

 In 2016, film critic Matt Zoller Seitz praised Prince's songs and music videos for "Batman", more so than the film itself, stating that his songs "suggest a
goofy, perverse, sensuous, somewhat introverted Batman film that so far we've never gotten from anyone", and arguing that Prince's music videos "are more
psychologically perceptive than any of the Batman films."
</item 3>

<item 4>
Howard Rosenman
Howard Rosenman (born February 1, 1945), also known as Zvi Howard Rosenman, is an American producer and motion picture executive.

 He specializes in producing romantic comedy films and documentary films.

 Some of his most popular productions include "Father of the Bride" (1991) starring Steve Martin and Diane Keaton, Joss Whedon's "Buffy the Vampire Slayer"
(1992) and "The Family Man" (2000) starring Nicolas Cage.

 Rosenman's documentary film "" won the Peabody Award and the 1990 Academy Award for Best Documentary Feature; his film "The Celluloid Closet" also won the
Peabody Award.
</item 4>

<item 5>
Christopher Nolan
Christopher Edward Nolan ( ; born 30 July 1970) is an English-American film director, producer, and screenwriter.

 He is one of the highest-grossing directors in history, and among the most successful and acclaimed filmmakers of the 21st century.
</item 5>

<item 6>
Batman vs. Two-Face
Batman vs. Two-Face is an upcoming 2017 American animated direct-to-video superhero film produced by Warner Bros.

 Animation and distributed by Warner Bros.

 Home Entertainment.

 It is a sequel to "".

 It will premiere at the New York Comic Con on October 8, 2017 and will be followed by a digital release on October 10, 2017 and on DVD and Blu-ray on October
17, 2017.

 Based on the 1960s "Batman" TV series, the film stars Adam West (in his final role before his death), Burt Ward and Julie Newmar reprising their roles of
Batman, Robin and Catwoman from the series.

 The film will show a tribute to the late Adam West.

 This could also be the final time Warner Bros.

 Animation would make a 1960s Batman film.
</item 6>

<item 7>
Rob Bottin
Robin R. Bottin (born April 1, 1959) is an American special make-up effects creator.

 Known for his collaborations with directors John Carpenter, Paul Verhoeven and David Fincher, Bottin worked with Carpenter on both "The Fog" and "The Thing",
with Verhoeven on "RoboCop", "Total Recall" and "Basic Instinct", and with Fincher on "Se7en" and "Fight Club".

 His other film credits include "Legend", "Innerspace" and "Fear and Loathing in Las Vegas".
</item 7>

<item 8>
Batman (score)
Batman: Original Motion Picture Score is the score album for the 1989 film "Batman" by Danny Elfman.

 According to the "Batman" DVD Special Edition, Elfman said that producer Jon Peters was not sure about him as a composer until Tim Burton made him play the
main titles.

 Elfman admitted he was stunned when Peters announced that the score would be released on its own album, as releasing a separate score album for a film was
something that was rarely done in the 1980s.

 Elfman's "The Batman Theme" went on to become an iconic piece.

 It served as the basis for the theme music of "", which premiered in 1992, although this was later changed.

 Some parts of the Elfman score are also heard in "", "" and "".

 Parts are also played in the queue, and on the station platform of Batman the Ride at various Six Flags theme parks.
</item 8>

<item 9>
Batman Forever (disambiguation)
Batman Forever is a 1995 film in the Batman film series.
</item 9>

<item 10>
Peter Guber
Howard Peter Guber (born March 1, 1942) is an executive, entrepreneur, educator, and author.

 He is Chairman and CEO of Mandalay Entertainment.

 Guber's most recent films from Mandalay Entertainment include "The Kids Are All Right", "Soul Surfer", and "Bernie".

 He has also produced "Batman", "The Witches of Eastwick", and "Flashdance."

 Guber's films have earned over $3 billion worldwide and 50 Academy Award nominations.
</item 10>
=== END ITEM id=c9ed24899457530c ===

=== ITEM id=7632f7953b4406d2 ===
Gold sentences and facts:
Sentence 0:  In 2003, he received the award for his actions during the Battle of Qala-i-Jangi, which took place in late November – early December 2001.
  Fact 0.0: In 2003, a person referred to as "he" received an award for his actions during the Battle of Qala-i-Jangi.
  Fact 0.1: The Battle of Qala-i-Jangi took place in late November to early December 2001.
Sentence 1: The Battle of Qala-i-Jangi (also incorrectly referred to as the "Battle of Mazar-i-Sharif") was a prisoner-of-war camp uprising that took place between November 25 and December 1, 2001, in northern Afghanistan, following the armed intervention by United States-led coalition forces to try to overthrow the Taliban's Islamic Emirate of Afghanistan, which it had accused of harboring al-Qaeda operatives.
  Fact 1.0: The Battle of Qala-i-Jangi is also incorrectly referred to as the "Battle of Mazar-i-Sharif".
  Fact 1.1: The Battle of Qala-i-Jangi was a prisoner-of-war camp uprising.
  Fact 1.2: The Battle of Qala-i-Jangi took place between November 25 and December 1, 2001.
  Fact 1.3: The Battle of Qala-i-Jangi took place in northern Afghanistan.
  Fact 1.4: The Battle of Qala-i-Jangi followed the armed intervention by United States-led coalition forces.
  Fact 1.5: The United States-led coalition forces intervened with armed forces to try to overthrow the Taliban's Islamic Emirate of Afghanistan.
  Fact 1.6: The coalition forces had accused the Taliban's Islamic Emirate of Afghanistan of harboring al-Qaeda operatives.

Retrieved items:
<item 1>
Orlov let
Orlov let ("Eagle's Flight") was the name of an anti-terrorist operation conducted by the Montenegrin police to arrest a group of Albanians who planned
terrorist attacks and an armed conflict in Albanian-inhabited parts of Montenegro.

 The group of 17 people planned operations from 2004 to their arrest on the night of 10/11 September 2006.

 Weapons and explosive material were found in police raids.

 The group's main operative was the destroying of cultural and religious buildings (Orthodox) in Albanian-inhabited territory.

 The group was organised by an Albanian association based in Detroit, and members of the Kosovo Liberation Army, from which they received financial aid and
smuggled arms into Montenegro.

 The group was given a total sentence of 51 years of prison.

 Relations between the state and the Albanian minority strained after the event.
</item 1>

<item 2>
Enemy Combatant (book)
Enemy Combatant is a memoir by British Muslim, Moazzam Begg, co-written by Victoria Brittain, former Associate Foreign Editor for "The Guardian", about Begg's
detention by the government of the United States of America in Bagram Detention Facility and at Camp Echo, Guantanamo Bay and his life prior to that detention.

 It was published in Britain as "Enemy Combatant: A British Muslim's Journey To Guantanamo and Back" (ISBN  ), and in the US as "Enemy Combatant: My
Imprisonment at Guantanamo, Bagram, and Kandahar" (ISBN  ).

 In the US, the foreword was written by David Ignatius of The Washington Post.
</item 2>

<item 3>
The Great War for Civilisation
The Great War for Civilisation: The Conquest of the Middle East is a book published in 2005 by the award-winning English journalist Robert Fisk.

 The book is a compilation of many of the articles Fisk wrote when he was serving as a correspondent in the Middle East for "The Times" and "The Independent".

 The book revolves around several key themes regarding the history of the modern Middle East: the Arab-Israeli conflict, the Soviet invasion of Afghanistan, the
Persian Gulf War as well as the 2003 Iraq War (United States invasion of Iraq) as well as other regional conflicts such as the Armenian Genocide and the
Algerian Civil War.

 The "Great War for Civilisation" is the second book Fisk has written about the Middle East with the first one, "Pity the Nation", (Nation Books, 2002) being
about the Lebanese Civil War.
</item 3>

<item 4>
Thomas Merton Award
The Thomas Merton Award has been awarded since 1972 by the Thomas Merton Center for Peace and Social Justice in Pittsburgh, United States.

 It is named after Thomas Merton and is given annually to "national and international individuals struggling for justice."
</item 4>

<item 5>
Soldiers of Egypt
Soldiers of Egypt (Ajnad Misr in Arabic) is an active Salafist Islamist militant group that has been operating near Cairo, Egypt.

 The groups was founded by Humam Muhammed in 2013, after he split away from the Ansar Bait al-Maqdis militant group.

 The group claims that its attacks are "retribution" for the August 2013 Rabaa Massacre; notably, the group targets only security forces.

 It has warned civilians of the presence of bombs that it has placed.

 The Cairo Court for Urgent Matters declared the group a terrorist group on 22 May 2014.

 It has been a Proscribed Organisation in the United Kingdom under the Terrorism Act 2000 since November 2014.

 The United States Department of State designated it a terrorist organization on 18 December 2014.
</item 5>

<item 6>
Mark E. Mitchell
Mark E. Mitchell is the current Acting Assistant Secretary of Defense for Special Operations and Low-Intensity Conflict of the United States Department of
Defense.

 A retired Colonel, Mitchell was the first member of the United States Army to be awarded the Distinguished Service Cross during the War in Afghanistan and was
the first to receive the award since the Vietnam War.

 In 2003, he received the award for his actions during the Battle of Qala-i-Jangi, which took place in late November – early December 2001.
</item 6>

<item 7>
Sayeed Salahudeen
Syed Mohammed Yusuf Shah, popularly known as Syed Salahudeen, is the head of Hizb-ul-Mujahideen, a terrorist group operating in Kashmir, and head of an alliance
of anti-India militant groups, the United Jihad Council, that works to annex the Indian state of Jammu and Kashmir to Pakistan .

 Salahuddin vowed to block any peaceful resolution to the Kashmir conflict, threatened to train more Kashmiri suicide bombers, and vowed to turn the Kashmir
valley “into a graveyard for Indian forces.”

 He is listed on the NIA Most Wanted list.

 On 26 June 2017, he was named as a Specially Designated Global Terrorist by the US Department Of State.

 After being designated as a Global terrorist, he addressed the Pakistani Media at Muzaffarabad’s Centre Press Club, claiming that "The declaration was a joint
move by the US, Israel, and India to express their animosity towards Pakistan."
</item 7>

<item 8>
Iraultza
Iraultza (Basque for Revolution) was a small Basque militant armed group of leftist tendency.

 Seven of the group would die preparing explosive devices.

 On June 27, 1986, a bomb claimed by Iraultza mangled a young worker called José Miguel Moros Peña in Portugalete.

 The victim would die in Cruces Hospital (Barakaldo) on August 13, 1986.

 The group is also known for being described by the United States Government as "probably [having] committed more bombings against American business interests
than any other European terrorist group".

 Soon after, the group, which had been active for about a decade with small attacks of explosives that never aimed to cause any personal injuries, announced its
dissolution.

 When the group was operational, it is estimated that it had approximately 20 members.
</item 8>

<item 9>
Battle of Qala-i-Jangi
The Battle of Qala-i-Jangi (also incorrectly referred to as the "Battle of Mazar-i-Sharif") was a prisoner-of-war camp uprising that took place between November
25 and December 1, 2001, in northern Afghanistan, following the armed intervention by United States-led coalition forces to try to overthrow the Taliban's
Islamic Emirate of Afghanistan, which it had accused of harboring al-Qaeda operatives.
</item 9>

<item 10>
Combat Terrorist Organization
Combat Terrorist Organization or Autonomous Combat Terrorist Organization (rus.

 Боевая Террористическая Организация) was a Russian radical Neo-Nazi terrorist group based in Saint Petersburg from 9 August 2003 to 2006.

 It was formed by two members of the "Mad Crowd" skinhead organization, Dmitriy Borovikov and Voevodin Alexey.
</item 10>
=== END ITEM id=7632f7953b4406d2 ===

=== ITEM id=4e69d2d0a84db22f ===
Gold sentences and facts:
Sentence 0: Essence is a monthly magazine for African American women between the ages of 18 and 49.
  Fact 0.0: Essence is a monthly magazine.
  Fact 0.1: Essence is for African American women.
  Fact 0.2: Essence is aimed at women between the ages of 18 and 49.
Sentence 1: ALT for Damerne (meaning "All for the Ladies" in English) is a Danish language weekly women's magazine published in Copenhagen, Denmark.
  Fact 1.0: ALT for Damerne is a women's magazine.
  Fact 1.1: ALT for Damerne means 'All for the Ladies' in English.
  Fact 1.2: ALT for Damerne is written in the Danish language.
  Fact 1.3: ALT for Damerne is a weekly magazine.
  Fact 1.4: ALT for Damerne is published in Copenhagen, Denmark.

Retrieved items:
<item 1>
PC Magazine
PC Magazine (shortened as PCMag) is an American computer magazine published by Ziff Davis.

 A print edition was published from 1982 to January 2009.

 Publication of online editions started in late 1994 and continues to this day.
</item 1>

<item 2>
Elle (magazine)
Elle is a worldwide lifestyle magazine of French origin that focuses on fashion, beauty, health, and entertainment.

 "Elle" is also the world's best-selling fashion magazine.

 It was founded by Pierre Lazareff and his wife Hélène Gordon in 1945.

 The title, in French, means "she" or "her".
</item 2>

<item 3>
Hänt Bild
Hänt Bild is a celebrity magazine published in Sweden.

 The magazine was started in 2003.

 It is part of Aller Media AB.

 It is published on a weekly basis.
</item 3>

<item 4>
Diva (magazine)
DIVA is a leading lesbian magazine in the United Kingdom, published monthly.

 It was launched in March 1994 by Millivres Ltd, under the editorship of Frances Williams and is now published by Twin Media Group, which is owned by Silke
Bader and Linda Riley.

 Silke Bader is also the publisher of Curve magazine in the USA and LOTL in Australia, and Linda Riley runs the European Diversity Awards and is the former
publisher of g3 magazine in the UK.
</item 4>

<item 5>
Exclaim!
Exclaim!

 is a monthly Canadian music magazine that features in-depth coverage of new music across all genres with a special focus on Canadian and cutting-edge artists.

 Content is based on the monthly print publication, which publishes 9 issues per year, distributing over 103,000 copies to over 2,600 locations across Canada.

 The magazine has an average of 361,200 monthly readers.

 Their website, exclaim.ca, has an average of 675,000 unique visitors a month.
</item 5>

<item 6>
The Fader
The Fader (stylized as The FADER) is a New York City-based music magazine launched in 1999 by Rob Stone and Jon Cohen, covering music, style and culture.

 It was the first print publication to be released on iTunes.
</item 6>

<item 7>
Essence Music Festival
The Essence Festival, known as "the party with a purpose", is an annual music festival which started in 1995 as a one-time event to celebrate the 25th
anniversary of "Essence", a magazine aimed primarily towards African-American women.

 It is the largest event celebrating African-American culture and music in the United States.

 Locally referred to as the Essence Fest, it has been held in New Orleans, Louisiana every year since 1994 except for 2006, when it was held in Houston, Texas
due to Hurricane Katrina's effect on New Orleans.

 It also held in Durban, KwaZulu-Natal in 2016.

 It features artists simultaneously performing on a main stage as well as four standing-room only superlounge stages.
</item 7>

<item 8>
Alt for Damerne
ALT for Damerne (meaning "All for the Ladies" in English) is a Danish language weekly women's magazine published in Copenhagen, Denmark.
</item 8>

<item 9>
Hänt Extra
Hänt Extra is a Swedish language weekly celebrity and youth magazine published by Aller Media in Stockholm, Sweden.
</item 9>

<item 10>
Essence (magazine)
Essence is a monthly magazine for African American women between the ages of 18 and 49.

 It is the only magazine that focuses on reaching an audience of black women, revolves around the black woman experience, and has remained for a long period of
time.

 The magazine covers fashion, lifestyle and beauty, with an intimate girlfriend-to-girlfriend tone, and its slogan "Fierce, Fun, and Fabulous" suggests the
magazine's goal of empowering African-American women.

 The topics the magazine discusses range from celebrities, to fashion, to point-of-view pieces addressing current issues in the African-American community.

 A number of its readers engage closely and personally with the publication, and it claims to be the magazine "for and about Black women".
</item 10>
=== END ITEM id=4e69d2d0a84db22f ===
