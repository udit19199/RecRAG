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

This file has 5 items.

=== ITEM id=6652995d28ab0c9f ===
Gold sentences and facts:
Sentence 0: Donald Trump is the 45th and current president , inaugurated on January 20 , 2017 .
  Fact 0.0: Donald Trump is the 45th president.
  Fact 0.1: Donald Trump is the current president.
  Fact 0.2: Donald Trump was inaugurated on January 20, 2017.
Sentence 1: President of the United States of America Presidential Seal Presidential Standard Incumbent Donald Trump since January 20 , 2017 Executive Branch of the U.S. Government Executive Office of the President Style Mr. President ( informal ) The Honorable ( formal ) His Excellency ( international correspondence ) Status Head of State Head of Government Member of Cabinet Domestic Policy Council National Economic Council National Security Council Residence White House Seat Washington , D.C. Appointer Electoral College of the United States Term length Four years , renewable once ( Amendment XXII ) Constituting instrument United States Constitution Inaugural holder George Washington Formation March 4 , 1789 ( 229 years ago ) ( 1789 - 03 - 04 ) Salary $400,000 annually Website whitehouse.gov
  Fact 1.0: The office of President of the United States of America is part of the Executive Branch of the U.S. Government.
  Fact 1.1: The incumbent is Donald Trump, since January 20, 2017.
  Fact 1.2: The office is part of the Executive Office of the President.
  Fact 1.3: The style of address is 'Mr. President' (informal), 'The Honorable' (formal), and 'His Excellency' (international correspondence).
  Fact 1.4: The office has the status of Head of State and Head of Government.
  Fact 1.5: The office is a member of the Cabinet, the Domestic Policy Council, the National Economic Council, and the National Security Council.
  Fact 1.6: The official residence is the White House.
  Fact 1.7: The seat is Washington, D.C.
  Fact 1.8: The appointer is the Electoral College of the United States.
  Fact 1.9: The term length is four years, renewable once (Amendment XXII).
  Fact 1.10: The constituting instrument is the United States Constitution.
  Fact 1.11: The inaugural holder is George Washington.
  Fact 1.12: The office was formed on March 4, 1789, which the page describes as 229 years ago.
  Fact 1.13: The salary is $400,000 annually.
  Fact 1.14: The official website is whitehouse.gov.

Retrieved items:
<item 1>
President of the United States
Kennedy ( 1961 -- 1963 ) Lyndon B. Johnson ( 1963 -- 1969 ) Richard Nixon ( 1969 -- 1974 ) Gerald Ford ( 1974 -- 1977 ) Jimmy Carter ( 1977 -- 1981 ) Ronald
Reagan ( 1981 -- 1989 ) George H.W. Bush ( 1989 -- 1993 ) Bill Clinton ( 1993 -- 2001 ) George W. Bush ( 2001 -- 2009 ) Barack Obama ( 2009 -- 2017 ) Donald
Trump ( 2017 -- present ) Presidency timelines Wilson Harding Coolidge Hoover F.D. Roosevelt Truman Eisenhower Kennedy L.B. Johnson Nixon Ford Carter Reagan
G.H.W. Bush Clinton G.W. Bush Obama Trump Book Category Chief executives of the United States Federal President of the United States State governors ( current
list ) Alabama Alaska Arizona Arkansas California list Colorado Connecticut Delaware Florida Georgia Hawaii list Idaho Illinois list Indiana list Iowa list
Kansas Kentucky list Louisiana Maine list Maryland list Massachusetts Michigan list Minnesota list Mississippi Missouri Montana Nebraska list Nevada New
Hampshire list New Jersey list New Mexico list New York list North Carolina list North Dakota list Ohio Oklahoma list Oregon
</item 1>

<item 2>
President of the United States
Administrative powers Suffice it to say that the President is made the sole repository of the executive powers of the United States , and the powers entrusted
to him as well as the duties imposed upon him are awesome indeed . William Rehnquist , Nixon v. General Services Administration , 433 U.S. 425 ( 1977 ) (
dissenting opinion ) The president is the head of the executive branch of the federal government and is constitutionally obligated to `` take care that the laws
be faithfully executed '' . The executive branch has over four million employees , including members of the military . Presidents make numerous executive branch
appointments : an incoming president may make up to 6,000 before taking office and 8,000 more while serving . Ambassadors , members of the Cabinet , and other
federal officers , are all appointed by a president with the `` advice and consent '' of a majority of the Senate . When the Senate is in recess for at least
ten days , the president may make recess appointments . Recess appointments are temporary and expire at the end of the next session of the Senate . The power of
a president to fire executive officials has long been a contentious political issue .
</item 2>

<item 3>
Vice President of the United States
Senate vice presidential bust collection In fiction Presidents actors Vice Presidents actors Candidates Line of succession U.S. Government Portal Biography
Portal Presidents of the United States Presidential line of succession in the United States of America Vice President ( Mike Pence ) Speaker of the House of
Representatives ( Paul Ryan ) President pro tempore of the Senate ( Orrin Hatch ) Secretary of State ( Mike Pompeo ) Secretary of the Treasury ( Steven Mnuchin
) Secretary of Defense ( Jim Mattis ) Attorney General ( Jeff Sessions ) Secretary of the Interior ( Ryan Zinke ) Secretary of Agriculture ( Sonny Perdue )
Secretary of Commerce ( Wilbur Ross ) Secretary of Labor ( Alex Acosta ) Secretary of Health and Human Services ( Alex Azar ) Secretary of Housing and Urban
Development ( Ben Carson ) Secretary of Transportation ( Elaine Chao ) Secretary of Energy ( Rick Perry ) Secretary of Education ( Betsy DeVos ) Secretary of
Veterans Affairs ( Robert Wilkie ) Secretary of Homeland Security ( Kirstjen Nielsen ) Jump up ^ Although Elaine Chao is the current Secretary of Transportation
, she is not a natural - born citizen ( acquired U.S. citizenship by naturalization ) and thus ineligible for the presidency .
</item 3>

<item 4>
Vice President of the United States
Roles of the vice president ( edit ) This article is part of a series on the Politics of the United States of America Federal Government ( show ) Constitution
of the United States Law Taxation Legislature ( show ) United States Congress House of Representatives Speaker Paul Ryan ( R ) Majority Leader Kevin McCarthy (
R ) Minority Leader Nancy Pelosi ( D ) Congressional districts United States Senate President Mike Pence ( R ) President Pro Tempore Orrin Hatch ( R ) President
Pro Tempore Emeritus Patrick Leahy ( D ) Majority Leader Mitch McConnell ( R ) Minority Leader Chuck Schumer ( D ) Executive ( show ) President of the United
States Donald Trump ( R ) Vice President of the United States Mike Pence ( R ) Cabinet Federal agencies Executive Office Judiciary ( show ) Supreme Court of the
United States Chief Justice John Roberts Kennedy Thomas Ginsburg Breyer Alito Sotomayor Kagan Gorsuch Courts of Appeals District Courts ( list ) Other tribunals
Elections ( show ) Presidential elections Midterm elections Off - year elections Political parties ( show ) Democratic Republican Third parties Federalism (
show ) State Government Governors Legislatures ( List ) State courts Local government Other countries Atlas This section needs additional citations for
verification .
</item 4>

<item 5>
President of the United States
President of the United States - wikipedia President of the United States Jump to : navigation , search `` POTUS '' redirects here . For the political talk
radio channel , see P.O.T.U.S. ( Sirius XM ) . For other uses , see President of the United States ( disambiguation ) . For a list , see List of Presidents of
the United States . President of the United States of America Presidential Seal Presidential Standard Incumbent Donald Trump since January 20 , 2017 Executive
Branch of the U.S. Government Executive Office of the President Style Mr. President ( informal ) The Honorable ( formal ) His Excellency ( international
correspondence ) Status Head of State Head of Government Member of Cabinet Domestic Policy Council National Economic Council National Security Council Residence
White House Seat Washington , D.C. Appointer Electoral College of the United States Term length Four years , renewable once ( Amendment XXII ) Constituting
instrument United States Constitution Inaugural holder George Washington Formation March 4 , 1789 ( 229 years ago ) ( 1789 - 03 - 04 ) Salary $400,000 annually
Website whitehouse.gov This article is part of a series on the Politics of the United
</item 5>

<item 6>
President of the United States
Images of documents written by U.S. presidents Presidents of the United States ( list ) George Washington ( 1789 -- 1797 ) John Adams ( 1797 -- 1801 ) Thomas
Jefferson ( 1801 -- 1809 ) James Madison ( 1809 -- 1817 ) James Monroe ( 1817 -- 1825 ) John Quincy Adams ( 1825 -- 1829 ) Andrew Jackson ( 1829 -- 1837 )
Martin Van Buren ( 1837 -- 1841 ) William Henry Harrison ( 1841 ) John Tyler ( 1841 -- 1845 ) James K. Polk ( 1845 -- 1849 ) Zachary Taylor ( 1849 -- 1850 )
Millard Fillmore ( 1850 -- 1853 ) Franklin Pierce ( 1853 -- 1857 ) James Buchanan ( 1857 -- 1861 ) Abraham Lincoln ( 1861 -- 1865 ) Andrew Johnson ( 1865 --
1869 ) Ulysses S. Grant ( 1869 -- 1877 ) Rutherford B. Hayes ( 1877 -- 1881 ) James A.
</item 6>

<item 7>
Vice President of the United States
Vice president of the United States - wikipedia Vice president of the United States Jump to : navigation , search For a list of office - holders , see List of
Vice Presidents of the United States . Vice President of the United States of America Vice Presidential seal Vice Presidential standard Incumbent Mike Pence
since January 20 , 2017 Executive branch of the U.S. government Office of the Vice President Style Mr. Vice President ( Informal ) The Honorable ( Formal ) Mr.
President ( When presiding over Senate ) His Excellency ( In international correspondence ) Member of Cabinet National Security Council President of the U.S.
Senate Residence Number One Observatory Circle Seat Washington , D.C. Appointer Electoral College Term length 4 years Constituting instrument U.S.
</item 7>

<item 8>
President of the United States
The President of the United States ( POTUS / ˈpoʊtəs / POH - təs ) is the head of state and head of government of the United States of America . The president
directs the executive branch of the federal government and is the commander - in - chief of the United States Armed Forces . In contemporary times , the
president is looked upon as the world 's most powerful political figure and as the leader of the only current global superpower . The role includes
responsibility for the world 's most expensive military that has the second largest nuclear arsenal . The president also leads the nation with the largest
economy by nominal GDP . The president possesses significant domestic and international hard and soft power . Article II of the Constitution establishes the
executive branch of the federal government . It vests the executive power of the United States in the president . The power includes the execution and
enforcement of federal law , alongside the responsibility of appointing federal executive , diplomatic , regulatory and judicial officers , and concluding
treaties with foreign powers with the advice and consent of the Senate .
</item 8>

<item 9>
President of the United States
The president is further empowered to grant federal pardons and reprieves , and to convene and adjourn either or both houses of Congress under extraordinary
circumstances . The president directs the foreign and domestic policies of the United States , and takes an active role in promoting his policy priorities to
members of Congress . In addition , as part of the system of checks and balances , Article One of the United States Constitution gives the president the power
to sign or veto federal legislation . Since the office of president was established in 1789 , its power has grown substantially , as has the power of the
federal government as a whole . Through the Electoral College , the registered voters indirectly elect the president and vice president to a four - year term .
This is the only federal election in the United States which is not decided by popular vote . Nine vice presidents became president by virtue of a president 's
intra-term death or resignation . The Twenty - second Amendment precludes any United States citizen from being elected president for a third term .
</item 9>

<item 10>
President of the United States
000 annually Website whitehouse.gov This article is part of a series on the Politics of the United States of America Federal Government ( show ) Constitution of
the United States Law Taxation Legislature ( show ) United States Congress House of Representatives Speaker Paul Ryan ( R ) Majority Leader Kevin McCarthy ( R )
Minority Leader Nancy Pelosi ( D ) Congressional districts United States Senate President Mike Pence ( R ) President Pro Tempore Orrin Hatch ( R ) President Pro
Tempore Emeritus Patrick Leahy ( D ) Majority Leader Mitch McConnell ( R ) Minority Leader Chuck Schumer ( D ) Executive ( show ) President of the United States
Donald Trump ( R ) Vice President of the United States Mike Pence ( R ) Cabinet Federal agencies Executive Office Judiciary ( show ) Supreme Court of the United
States Chief Justice John Roberts Kennedy Thomas Ginsburg Breyer Alito Sotomayor Kagan Gorsuch Courts of Appeals District Courts ( list ) Other tribunals
Elections ( show ) Presidential elections Midterm elections Off - year elections Political parties ( show ) Democratic Republican Third parties Federalism (
show ) State Government Governors Legislatures ( List ) State courts Local government Other countries Atlas The President of the United States ( POTUS / ˈpoʊtəs
/
</item 10>
=== END ITEM id=6652995d28ab0c9f ===

=== ITEM id=34f281f161a64c43 ===
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
Drawing
The wide availability of drawing instruments makes drawing one of the most common artistic activities . In addition to its more artistic forms , drawing is
frequently used in commercial illustration , animation , architecture , engineering and technical drawing . A quick , freehand drawing , usually not intended as
a finished work , is sometimes called a sketch . An artist who practices or works in technical drawing may be called a drafter , draftsman or a draughtsman .
Contents ( hide ) 1 Overview 2 History 2.1 Drawing in communication 2.2 Drawing in manuscripts 2.3 Drawing in science 2.4 Drawing as artistic expression 3
Notable artists and draftsmen 4 Materials 5 Technique 6 Tone 7 Form and proportion 8 Perspective 9 Artistry 10 Process 11 Seven Ps of drawing 12 See also 13
References 14 External links Overview ( edit ) Galileo Galilei . Phases of the Moon . 1616 . Drawing is one of the major forms of expression within the visual
arts . It is generally concerned with the marking of lines and areas of tone onto paper / other material , where the accurate representation of the visual world
is expressed upon a plane surface .
</item 1>

<item 2>
Drawing
Sometimes the artist leaves a section of the image untouched while filling in the remainder . The shape of the area to preserve can be painted with masking
fluid or cut out of a frisket and applied to the drawing surface , protecting the surface from stray marks until the mask is removed . Another method to
preserve a section of the image is to apply a spray - on fixative to the surface . This holds loose material more firmly to the sheet and prevents it from
smearing . However the fixative spray typically uses chemicals that can harm the respiratory system , so it should be employed in a well - ventilated area such
as outdoors . Another technique is subtractive drawing in which the drawing surface is covered with graphite or charcoal and then erased to make the image .
Tone ( edit ) A pencil drawing with hatching and shading Shading is the technique of varying the tonal values on the paper to represent the shade of the
material as well as the placement of the shadows . Careful attention to reflected light , shadows and highlights can result in a very realistic rendition of the
image . Blending uses an implement to soften or spread the original drawing strokes .
</item 2>

<item 3>
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
Drawing
Jump up ^ Ruskin , John ( 1857 ) . The Elements of Drawing . Mineola , NY : Dover Publishcations Inc . Jump up ^ McManus , I.C. ; Chamberlain , R.S. ; Loo , P.
- K. ; Rankin , Q. ; Riley , H. ; Brunswick , N. ( 2010 ) . `` Art students who can not draw : exploring the relations between drawing ability , visual memory ,
accuracy of copying , and dyslexia '' ( PDF ) . Psychology of Aesthetics , Creativity and the Arts . 4 : 18 -- 30 . doi : 10.1037 / a0017335 . Jump up ^ Fayena
- Tawil , F. ; Kozbelt , A. ; Sitaras , S. ( 2011 ) . `` Think global , act local : A protocol analysis comparison of artists ' and nonartists ' cognitions ,
metacognitions , and evaluations while drawing '' . Psychology of Aesthetics , Creativity and the Arts . 5 : 135 -- 45 .
</item 5>

<item 6>
Drawing
Traditional drawings were monochrome , or at least had little colour , while modern colored - pencil drawings may approach or cross a boundary between drawing
and painting . In Western terminology , drawing is distinct from painting , even though similar media often are employed in both tasks . Dry media , normally
associated with drawing , such as chalk , may be used in pastel paintings . Drawing may be done with a liquid medium , applied with brushes or pens . Similar
supports likewise can serve both : painting generally involves the application of liquid paint onto prepared canvas or panels , but sometimes an underdrawing is
drawn first on that same support . Drawing is often exploratory , with considerable emphasis on observation , problem - solving and composition . Drawing is
also regularly used in preparation for a painting , further obfuscating their distinction . Drawings created for these purposes are called studies Madame
Palmyre with Her Dog , 1897 . Henri de Toulouse - Lautrec There are several categories of drawing , including figure drawing , cartooning , doodling , and free
hand .
</item 6>

<item 7>
Drawing
Drawing - wikipedia Drawing Jump to : navigation , search For other uses , see Drawing ( disambiguation ) . Pen and wash lion by Rembrandt in the Louvre Drawing
is a form of visual art in which a person uses various drawing instruments to mark paper or another two - dimensional medium . Instruments include graphite
pencils , pen and ink , inked brushes , wax colored pencils , crayons , charcoal , chalk , pastels , various kinds of erasers , markers , styluses , various
metals ( such as silverpoint ) and electronic drawing . A drawing instrument releases a small amount of material onto a surface , leaving a visible mark . The
most common support for drawing is paper , although other materials , such as cardboard , plastic , leather , canvas , and board , may be used . Temporary
drawings may be made on a blackboard or whiteboard or indeed almost anything . The medium has been a popular and fundamental means of public expression
throughout human history . It is one of the simplest and most efficient means of communicating visual ideas . The wide availability of drawing instruments makes
drawing one of the most common artistic activities .
</item 7>

<item 8>
Drawing
Escher Arshile Gorky George Grosz Paul Klee Oscar Kokoschka Käthe Kollwitz Alfred Kubin André Masson Alphonse Mucha Jules Pascin Pablo Picasso Egon Schiele Jean
- Michel Basquiat Andy Warhol Frank Auerbach 21st : Tracey Emin Banksy Materials ( edit ) The medium is the means by which ink , pigment or color are delivered
onto the drawing surface . Most drawing media are either dry ( e.g. graphite , charcoal , pastels , Conté , silverpoint ) , or use a fluid solvent or carrier (
marker , pen and ink ) . Watercolor pencils can be used dry like ordinary pencils , then moistened with a wet brush to get various painterly effects . Very
rarely , artists have drawn with ( usually decoded ) invisible ink . Metalpoint drawing usually employs either of two metals : silver or lead . More rarely used
are gold , platinum , copper , brass , bronze , and tinpoint . Paper comes in a variety of different sizes and qualities , ranging from newspaper grade up to
high quality and relatively expensive paper sold as individual sheets .
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
Drawing
There are also many drawing methods , such as line drawing , stippling , shading , the surrealist method of entopic graphomania ( in which dots are made at the
sites of impurities in a blank sheet of paper , and lines are then made between the dots ) , and tracing ( drawing on a translucent paper , such as tracing
paper , around the outline of preexisting shapes that show through the paper ) . A quick , unrefined drawing may be called a sketch . In fields outside art ,
technical drawings or plans of buildings , machinery , circuitry and other things are often called `` drawings '' even when they have been transferred to
another medium by printing . History ( edit ) Drawing in communication ( edit ) Drawing is one of the oldest forms of human expression , with evidence for its
existence preceding that of written communication . It is believed that drawing was used as a specialised form of communication before the invention of the
written language , demonstrated by the production of cave and rock paintings around 30,000 years ago ( Art of the Upper Paleolithic ) . These drawings , known
as pictograms , depicted objects and abstract concepts .
</item 10>
=== END ITEM id=34f281f161a64c43 ===

=== ITEM id=450d24a20f9d403c ===
Gold sentences and facts:
Sentence 0: The Burzahom site is a prehistoric settlement in the village of the same name in the Srinagar District . It is 16 kilometres ( 9.9 mi ) to the northeast of Srinagar on the Naseem - Shalimar road . The elevation of the site is 1,800 metres ( 5,900 ft ) above sea - level . It is the northernmost excavated Neolithic site of India . The site is on an ancient Pleistocene lake bed . The location is in a high terrace which is part of the floodplain of the Jhelum river and has Karewa soil ( clay ) formation . The site has a commanding view of the Dal lake which is about 2 kilometres ( 1.2 mi ) away . In the Kashmiri language ' Burzahom ' means `` birch '' , a tree species ( that generally grows in the elevation range of 3,000 to 4,200 metres ( 9,800 to 13,800 ft ) in the Himalayas ) , which is found in the excavated housing area in the form of roofing material , and thus confirming the existence of the tree even in the pre-historic Neolithic times .
  Fact 0.0: The Burzahom site is a prehistoric settlement.
  Fact 0.1: The Burzahom site is in the village of the same name, in the Srinagar District.
  Fact 0.2: The Burzahom site is 16 kilometres (9.9 mi) northeast of Srinagar.
  Fact 0.3: The Burzahom site is on the Naseem-Shalimar road.
  Fact 0.4: The elevation of the Burzahom site is 1,800 metres (5,900 ft) above sea level.
  Fact 0.5: The Burzahom site is the northernmost excavated Neolithic site of India.
  Fact 0.6: The Burzahom site is on an ancient Pleistocene lake bed.
  Fact 0.7: The Burzahom site's location is in a high terrace.
  Fact 0.8: The high terrace is part of the floodplain of the Jhelum river.
  Fact 0.9: The location has Karewa soil (clay) formation.
  Fact 0.10: The Burzahom site has a commanding view of the Dal lake.
  Fact 0.11: The Dal lake is about 2 kilometres (1.2 mi) away from the site.
  Fact 0.12: In the Kashmiri language, 'Burzahom' means 'birch'.
  Fact 0.13: Birch is a tree species that generally grows in the elevation range of 3,000 to 4,200 metres (9,800 to 13,800 ft) in the Himalayas.
  Fact 0.14: Birch is found in the excavated housing area of the Burzahom site in the form of roofing material.
  Fact 0.15: The presence of birch roofing material confirms the existence of the tree even in pre-historic Neolithic times.

Retrieved items:
<item 1>
Burzahom archaeological site
The pottery found at the site were in an early stage of hand crafting , of the coarse variety , in steel - grey , dull red , brown , and buff colours with mat
prints at the bottom ; they were in the shape of bowl , vase and stem . The antiquities did not reveal any signs of burials sites . Late Kot - Diji type pots
were found belonging to Period Ib . Period II ( edit ) In the Period II , the finds excavated revealed that people had moved out from pit dwelling to structures
built at the ground level . However , the pits and its associated chambers formed the base floor of the superstructure , which was made up by filling the pits
and covering it with mud plaster , and occasionally painted in red ochre . Post-holes around the pits revealed that the superstructures were made of wood built
over compacted Karewa soil floors . This period also brought out , for the first time , the burial customs of the Neolithic people . Both human and animal
skeletons were found in deep oval shaped pits , located either below the floors of the dwelling units or in its precincts . These pits were filled with ash ,
stones and potsherds .
</item 1>

<item 2>
Burzahom archaeological site
An interesting find of this period is of two standalone finished flat stone slabs . The carving on one is not distinct . The other stone slab is 48 -- 27
centimetres ( 19 -- 11 in ) which depicts , on one polished side , sketches of hunting scenes such as a hunter spearing ( with a Ker ) an antlered deer and
another hunter in the process of releasing an arrow , and a sketch of the movement of the Sun , at two levels . The carved figures are distinctly visible .
Agricultural practices were noted during the Periods I and II and crops grown were inferred as wheat , barley and lentil ; finding lentils established a link of
the Neolithic people with Central Asia , crossing over the Himalayas . The people who resided here were characterized as `` long headed dolichocranic '' . Two
female skulls , different from the male skulls , were also reported . The finds did not indicate of any external ethnic intrusions during the entire Neolithic
period but showed more affinity to the Harappan people . Burzahom represents the southernmost extent of what is known as Northern Neolithic culture of Asia .
</item 2>

<item 3>
Burzahom archaeological site
Khazanchi and his team of ASI from 1960 and 1971 The Burzahom archaeological site is located in the Kashmir Valley of the Indian state of Jammu and Kashmir .
Archaeological excavations have revealed four phases of cultural significance between 3000 BC and 1000 BC . Periods I and II represent the Neolithic era ;
Period IlI the Megalithic era ( of massive stone menhirs and wheel turned red pottery ) ; and Period IV relates to the early Historical Period ( Post-megalithic
period ) . The findings , recorded in stratified cultural deposits representing prehistoric human activity in Kashmir , are based on detailed investigations
that cover all aspects of the physical evidence of the site , including the ancient flora and fauna . The Burzahom site revealed the transition from the
subterranean and ground level housing features of the Neolithic people to the mudbrick structures of the Megalithic people . The large cache of tools and
implements made of bone and stone found at the site shows that the inhabitants were hunting and farming .
</item 3>

<item 4>
Burzahom archaeological site
Burzahom Archaeological site - wikipedia Burzahom Archaeological site Jump to : navigation , search Burzahom archaeological site برزۂہوم A pot excavated from
Burzahom Shown within India Location Srinagar , Jammu and Kashmir Coordinates 34 ° 10 ′ N 73 ° 54 ′ E ﻿ / ﻿ 34.167 ° N 73.900 ° E ﻿ / 34.167 ; 73.900
Coordinates : 34 ° 10 ′ N 73 ° 54 ′ E ﻿ / ﻿ 34.167 ° N 73.900 ° E ﻿ / 34.167 ; 73.900 Type Pre-historic Settlement History Periods Neolithic , Megalithic and
early Historic period . Site notes Excavation dates 1939 and from 1960 to 1971 Archaeologists de Terra and Patersonof Yale - Cambridge Expedition in 1939 and
T.N.
</item 4>

<item 5>
Burzahom archaeological site
These pits were filled with ash , stones and potsherds . Some of the human skulls found here had trepanning ( bored hole ) marks . In many pits , bones of dogs
and antlered deer were found along with human skeletons . The skeletons of humans were found in the burial pits in a sitting position along with bones of
animals . Pottery finds showed better finish compared to the earlier Period I . The pots were of polished black ware , mostly handmade , in the form of a dish
with stand , a high - necked jar , and so forth . Also found was a wheel turned red ware pot which contained 950 beads made from carnelian and agate ( inferred
as items for sale ) , which was thought to belong to the later part of this period . A very impressive painted pottery ware recovered from this period was a
globular red ware pot made on a turntable ; the painting on the pot was of a wild goat of black colour with long horns and hanging ears . Another pottery item
which is of interest is a polished black ware in globular shape jar with a long neck and flaring mouth .
</item 5>

<item 6>
Burzahom archaeological site
Period III ( edit ) Some Megalithic Period Menhirs are next to Neolithic pits , suggesting a gradual transition between the two phases . The Menhirs , boulders
formed due the varying temperatures , were brought down from the hills with great effort by the people and installed to mark notable events of the community .
These are rough in shape , huge and of considerable weight and height , and are `` free - standing '' . Craftsmanship was superior during this period with finds
of wheel made durable hard red ware , copper objects , and tools made of bone and stone . Structures made of rubble were also found . Finds of a few copper
arrowheads indicated knowledge of metallurgy . Period IV ( edit ) Period IV ( dated to the 3rd -- 4th century AD ) , the last phase of human occupation at
Burzahom , was related to the early Historical Period . The structures built were superior compared to the earlier period , and were made from mud - bricks .
Pottery was also superior , of red ware type with slips and wheel turned . Some iron antiquaries were also found .
</item 6>

<item 7>
Burzahom archaeological site
These are : Periods I and II of the Neolithic ( Period I is called aceramic and Period II is called ceramic ) origin , particularly characterized by dwelling
pits ( the largest measuring 2.74 metres ( 9 ft 0 in ) at the top to 4.75 metres ( 15.6 ft ) at the base at a depth of 3.95 metres ( 13.0 ft ) ) ; Period III of
the Megalithic sequence noted by the free standing large stone Menhirs installed at the site by shifting boulders manually from the hills ; and Period IV of the
early Modern Period . The skeletal remains of the Neolithic humans found at Burzahom are similar to those found in Harappa of the Indus Valley Civilization .
Burzahom 's ceramic industry was mostly of hunting based culture and is different from the Chinese Neolithic pottery . The economy of the people was found to be
based on hunting and gathering with a nascent stage of cultivation practices . Pottery made in Burzahom showed close affinity to those found in the Swat valley
in Pakistan , particularly in respect of its shapes and decorations of the black ware pottery .
</item 7>

<item 8>
Burzahom archaeological site
The unearthed Antiquities ( of art , architecture , customs and rituals ) indicate that the prehistoric people of the Burzahom established contact with Central
Asia and South West Asia and also had links to the Gangetic plains and peninsular India . The interaction of local and foreign influences is demonstrated by the
art , architecture , customs , rituals and language demonstrated by some engravings on pottery and other artifacts . Some historians have stated that the Vedic
Aryan culture extended into Kashmir , but archaeological investigation at Burzahom does not support the theory . Contents ( hide ) 1 Location 2 History 3
Discovery 3.1 Period I 3.2 Period II 3.3 Period III 3.4 Period IV 4 Preservation 5 See also 6 References 7 Bibliography Location ( edit ) The Burzahom site is a
prehistoric settlement in the village of the same name in the Srinagar District . It is 16 kilometres ( 9.9 mi ) to the northeast of Srinagar on the Naseem -
Shalimar road .
</item 8>

<item 9>
Burzahom archaeological site
The burial practices and type of tools recovered from the site were inferred as having close resemblance to those found in the North Chinese Neolithic culture .
Period I ( edit ) The remarkable find during this period was of pits which were inferred as dwelling units ; these were in circular or oval shape dug in compact
natural Karewa soil formation . Some of the deep pits had steps and ladder access to the bottom level . In some of the pits the stratification revealed ash and
charcoal layers , which denoted human occupancy . Post holes on the sides of pits at the surface level denoted the presence of superstructures covered with
thatch made of birch . Shallow pits of circular shape of 60 -- 91 centimetres ( 24 -- 36 in ) diameter adjoining the housing pits were found to contain bones of
animals and also tools made of bones ( of antlers used for making tools ) and stones ( harpoons , needles with or without eyes , awls ) . Carbon dating
established that the Neolithic culture of this site was traceable to the 3rd millennium BC , the earliest occupation at the site was dated to before 2,357 BC .
</item 9>

<item 10>
Burzahom archaeological site
Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use
and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 10>
=== END ITEM id=450d24a20f9d403c ===

=== ITEM id=ef006750e790e17f ===
Gold sentences and facts:
Sentence 0: A third series was confirmed to be in production by writer Simon Nye at the BFI and Radio Times Television Festival on 8 April 2017 . He described the third series as having `` some exotic new animals '' , and that production would begin in three weeks ' time , upon Keeley Hawes 's arrival in Corfu for filming . It will be set in 1937 .
  Fact 0.0: A third series was confirmed to be in production by writer Simon Nye.
  Fact 0.1: Simon Nye confirmed the third series at the BFI and Radio Times Television Festival on 8 April 2017.
  Fact 0.2: Simon Nye described the third series as having some exotic new animals.
  Fact 0.3: Simon Nye said production of the third series would begin in three weeks' time.
  Fact 0.4: Production of the third series would begin upon Keeley Hawes's arrival in Corfu for filming.
  Fact 0.5: The third series will be set in 1937.

Retrieved items:
<item 1>
The Durrells
Jump up ^ Huntman , Ruth ( 26 March 2016 ) . `` Gerald Durrell was my hero ... I married him for his zoo '' . The Guardian . Retrieved April 26 , 2017 . Jump up
^ `` SUNNIER CLIMES . Where is The Durrells filmed ? Corfu locations and Mediterranean settings for the ITV drama '' . The Sun . 28 May 2017 . Retrieved 27
December 2017 . Jump up ^ Travis , Ben ( 23 April 2017 ) . `` The Durrells , Series 2 : cast , locations , and three other things to know as the Keeley Hawes
drama returns '' . London Evening Standard . Retrieved 27 December 2017 . ^ Jump up to : `` BARB - Weekly Top 10 Programmes '' . Broadcasters ' Audience
Research Board . Retrieved 9 May 2016 . Jump up ^ `` BARB - Weekly Top 30 Programmes '' . Broadcasters ' Audience Research Board . Retrieved 9 May 2016 .
</item 1>

<item 2>
The Durrells
The Durrells - wikipedia The Durrells Jump to : navigation , search The Durrells Genre Comedy - drama Based on The Corfu Trilogy by Gerald Durrell Written by
Simon Nye Theme music composer Ruth Barrett Country of origin United Kingdom Original language ( s ) English No. of series No. of episodes 12 Production
Executive producer ( s ) Lee Morris Sally Woodward Gentle Simon Nye Producer ( s ) Christopher Hall Cinematography Julian Court , James Aspinall Running time 46
-- 47 minutes Production company ( s ) Sid Gentle Films Masterpiece co-production Release Original network ITV Picture format 16 : 9 1080i Audio format Stereo
Original release 3 April 2016 ( 2016 - 04 - 03 ) -- present External links Website The Durrells ( also known as The Durrells in Corfu on American television )
is a British comedy - drama series based on Gerald Durrell 's three autobiographical books about his family 's four years ( 1935 -- 1939 ) on the Greek Island
of Corfu , which began airing on 3 April 2016 .
</item 2>

<item 3>
The Durrells
Jump up ^ The episode 's world premiere date was at the BFI and Radio Times Television Festival on 8 April 2017 . Series 3 ( 2018 ) ( edit ) ( 1937 ) A third
series was confirmed to be in production by writer Simon Nye at the BFI and Radio Times Television Festival on 8 April 2017 . He described the third series as
having `` some exotic new animals '' , and that production would begin in three weeks ' time , upon Keeley Hawes 's arrival in Corfu for filming . It will be
set in 1937 . Reception ( edit ) Reception to the first episode was positive , with Gerard O'Donovan ( The Telegraph ) calling it `` a series that 's not only
sun - drenched and liberating , but also catches its source material 's high good humour without labouring it and weaves an authentic sense of the innocent
exoticism of the original , '' before awarding it four stars .
</item 3>

<item 4>
The Durrells
Season Episodes Season Premiere Season Finale 6 3 April 2016 ( 2016 - 04 - 03 ) 8 May 2016 ( 2016 - 05 - 08 ) 6 23 April 2017 ( 2017 - 04 - 23 ) 28 May 2017 (
2017 - 05 - 28 ) Episodes ( edit ) Series 1 ( 2016 ) ( edit ) ( 1935 ) No . overall No. in series Title Directed by Written by Original air date UK viewers (
millions ) `` Episode 1 '' Steve Barron Simon Nye 3 April 2016 ( 2016 - 04 - 03 ) 8.81 The island of Corfu 's distinct contrast with England is what enthrals
Bournemouth mother Louisa Durrell to uproot her disenchanted family from a life of no prospects and venture into one with a lessened need for conformity .
However , the appeal becomes ever distant to see upon their arrival .
</item 4>

<item 5>
The Durrells
Jump up ^ `` Twitter - ITV Media '' . Twitter . Retrieved 11 April 2016 . Jump up ^ `` Digital Spy - Forums - The Ratings Thread ( Part 67 ) '' . Digital Spy
Forums . Retrieved 11 April 2016 . Jump up ^ `` BARB - Weekly Top 10 Programmes ( w / e 3rd April 2016 ) '' . Broadcasters ' Audience Research Board . Retrieved
11 April 2016 . Jump up ^ `` ITV Media - Press Releases - `` ITV recommissions hit drama The Durrells '' `` . ITV Media . Retrieved 16 April 2016 . Jump up ^ ``
ITV Press Centre - `` More Sun , Sea and Animals as the cast of The Durrells return to Corfu to film the second series of the ITV drama '' `` . ITV Press Centre
. ITV . Retrieved 12 August 2016 . Jump up ^ `` IMDb - The Durrells '' . Internet Movie Database . Retrieved 12 August 2016 .
</item 5>

<item 6>
The Durrells
See also ( edit ) Gerald Durrell 's Corfu trilogy books : My Family and Other Animals ( 1956 ) Birds , Beasts , and Relatives ( 1969 ) The Garden of the Gods (
1978 ) My Family and Other Animals , a 1987 BBC series based on the Corfu trilogy by Durrell My Family and Other Animals , a 2005 BBC telemovie based on the
1956 book by Durrell References ( edit ) Jump up ^ `` Early life of Gerald Durrell comes to TV screens soon '' . Jersey Evening Post . 24 March 2016 . Retrieved
1 April 2016 . Jump up ^ `` ITV commissions new six - part drama The Durrells '' . ITV Press Centre . 18 June 2015 . Retrieved 1 April 2016 . Jump up ^ `` The
Durrells ( TV Series 2016 -- ) '' . Internet Movie Database . Retrieved 1 April 2016 . Jump up ^ Huntman , Ruth ( 26 March 2016 ) .
</item 6>

<item 7>
The Durrells
The show and its episodes attained high ratings on IMDb , around 8 / 10 . BROADCAST ( edit ) Internationally , the series was acquired in Australia by the Seven
Network and premiered on 24 August 2016 . In the United States , PBS began airing the show , retitled as The Durrells in Corfu , on 16 October 2016 at 8pm . The
Durrells started screening in New Zealand on 26 October 2016 on Prime TV . In Canada , the Canadian Broadcasting Corporation began airing Series 1 on 13
September 2017 . In Spain , the series has been acquired by the streaming platform Filmin . A third series of The Durrells was announced by writer Simon Nye at
the BFI and Radio Times Television Festival on 8 April 2017 . It began filming in May 2017 and is expected to air on ITV in the Spring of 2018 .
</item 7>

<item 8>
The Durrells
The series is written by Simon Nye , directed by Steve Barron and Roger Goldby , and produced by Christopher Hall . The executive producers are Lee Morris and
Sally Woodward Gentle. Lee Durrell , Gerald Durrell 's widow and director of the Durrell Wildlife Conservation Trust , acted as consultant . The series is
partly filmed on Corfu and at Ealing Studios in London . Contents ( hide ) 1 Story 2 Characters 3 Series overview 4 Episodes 4.1 Series 1 ( 2016 ) 4.2 Series 2
( 2017 ) 4.3 Series 3 ( 2018 ) 5 Reception 6 Broadcast 7 See also 8 References 9 External links Story ( edit ) The series begins in 1935 , when Louisa Durrell
suddenly announces that she and her four children will move from Bournemouth to the Greek island of Corfu . Her husband has died some years earlier and the
family is experiencing financial problems . A Homeric battle ensues as the family adapts to life on the island which , despite a lack of electricity , is cheap
and an earthly paradise .
</item 8>

<item 9>
The Durrells
Internet Movie Database . Retrieved 12 August 2016 . Jump up ^ Knox , David ( 30 June 2016 ) . `` Seven picks up UK drama The Durrells '' . TV Tonight .
Retrieved 27 August 2016 . Jump up ^ Dunk , Tiffany ( 24 August 2016 ) . `` Seven takes punt that The Durrells can recreate Downton Abbey 's huge success '' .
news.com.au . Retrieved 27 August 2016 . Jump up ^ `` Screener - TV Listings - `` The Durrells in Corfu on Masterpiece '' `` . 8 October 2016 . Retrieved 8
October 2016 . Jump up ^ `` CBC ANNOUNCES FALL 2017 BROADCAST PREMIERE DATES FOR NEW AND RETURNING TELEVISION SERIES '' . CBC Media Centre . Jump up ^ `` Los
Durrell '' . Retrieved 27 September 2017 . Jump up ^ 28 day data : 7,247,000 on ITV , 1,218,000 on ITV HD and 343,
</item 9>

<item 10>
The Durrells
Characters ( edit ) Main Keeley Hawes as Louisa Durrell Milo Parker as Gerry Durrell Josh O'Connor as Larry Durrell Daisy Waterstone as Margo Durrell Callum
Woodhouse as Leslie Durrell Lucy Black as Florence Petrides Yorgos Karamihos as Dr. Theo Stephanides Alexis Georgoulis as Spiros Halikiopoulos Leslie Caron as
Countess Mavrodaki Ulric von der Esch as Sven Lundblad Anna Savva as Lugaretzia Errika Bigiou as Vasilia ( series 2 ) Recurring James Cosmo as Captain Creech
Maximilian Befort as Max Manolis Emmanouel as Sotos Andrew Bicknell as Headmaster Graham Seed as Mr Trevitt Liz Watts as Nancy Ben Hall as Donald Hara - Joy
Ermidi as Alexia Yorgos Tryfonas as Market Trader Nick Orestis Chaniotakis as Monk Arsenis Grimmas as Guest House Manager Spiros Kasfikis as Durrell 's House
Owner Daniel Lapaine as Hugh Jarvis Series overview ( edit ) Season Episodes Season Premiere Season Finale 6 3 April 2016 ( 2016 -
</item 10>
=== END ITEM id=ef006750e790e17f ===

=== ITEM id=0e180567056a5dd4 ===
Gold sentences and facts:
Sentence 0: The ABA Model Rules of Professional Conduct , created by the American Bar Association ( ABA ) , are a set of rules that prescribe baseline standards of legal ethics and professional responsibility for lawyers in the United States . They were promulgated by the ABA House of Delegates upon the recommendation of the Kutak Commission in 1983 . The rules are merely recommendations , or models , ( hence the name `` Model Rules '' ) and are not themselves binding . However , having a common set of Model Rules facilitates a common discourse on legal ethics , and simplifies professional responsibility training as well as the day - to - day application of such rules . As of 2015 , 49 states and four territories have adopted the rules in whole or in part , of which the most recent to do so was the Commonwealth of the Northern Mariana Islands in March 2015 . California is the only state that has not adopted the ABA Model Rules , while Puerto Rico is the only U.S. jurisdiction outside of confederation has not adopted them but instead has its own Código de Ética Profesional .
  Fact 0.0: The ABA Model Rules of Professional Conduct were created by the American Bar Association (ABA).
  Fact 0.1: The ABA Model Rules of Professional Conduct are a set of rules.
  Fact 0.2: The ABA Model Rules prescribe baseline standards of legal ethics for lawyers in the United States.
  Fact 0.3: The ABA Model Rules prescribe baseline standards of professional responsibility for lawyers in the United States.
  Fact 0.4: The ABA House of Delegates promulgated the rules upon the recommendation of the Kutak Commission.
  Fact 0.5: The Kutak Commission's recommendation came in 1983.
  Fact 0.6: The rules are merely recommendations, or models.
  Fact 0.7: The name Model Rules reflects that the rules are recommendations or models.
  Fact 0.8: The rules are not themselves binding.
  Fact 0.9: Having a common set of Model Rules facilitates a common discourse on legal ethics.
  Fact 0.10: Having a common set of Model Rules simplifies professional responsibility training.
  Fact 0.11: Having a common set of Model Rules simplifies the day-to-day application of such rules.
  Fact 0.12: As of 2015, 49 states and four territories had adopted the rules in whole or in part.
  Fact 0.13: The most recent jurisdiction to adopt the rules as of 2015 was the Commonwealth of the Northern Mariana Islands.
  Fact 0.14: The Commonwealth of the Northern Mariana Islands adopted the rules in March 2015.
  Fact 0.15: California is the only state that has not adopted the ABA Model Rules.
  Fact 0.16: Puerto Rico is the only U.S. jurisdiction outside of confederation that has not adopted the ABA Model Rules.
  Fact 0.17: Puerto Rico has its own Codigo de Etica Profesional instead.

Retrieved items:
<item 1>
American Bar Association Model Rules of Professional Conduct
As of 2015 , 49 states and four territories have adopted the rules in whole or in part , of which the most recent to do so was the Commonwealth of the Northern
Mariana Islands in March 2015 . California is the only state that has not adopted the ABA Model Rules , while Puerto Rico is the only U.S. jurisdiction outside
of confederation has not adopted them but instead has its own Código de Ética Profesional . Contents ( hide ) 1 State , commonwealth and territorial adoptions
of the Rules 1.1 California 2 Professional regulation 3 See also 4 References 5 External links State , Commonwealth and territorial adoptions of the Rules (
edit ) The American Bar Association is a private sector voluntary professional association . As such , it has no lawmaking power or regulatory authority , but
it is just like any other nongovernmental bar association . Accordingly , the Model Rules are not legally binding on ABA members or within any particular
jurisdiction .
</item 1>

<item 2>
American Bar Association Model Rules of Professional Conduct
However , the Model Rules have been adopted , in whole or in part , and sometimes in variation , as the rules of professional conduct for attorneys in 49 states
, the District of Columbia , and four of the five inhabited U.S. territories . The rules of professional conduct or professional responsibility adopted in a
particular state are legally enforceable against the lawyers of that state as well as any lawyer practicing there on a pro hac vice basis . On December 17 ,
2008 , New York announced that it would finally abandon the old ABA Model Code of Professional Responsibility ( the last state to do so ) and adopt a heavily
modified version of the ABA Model Rules , effective April 1 , 2009 . New York 's version of the Model Rules was created by adjusting the standard Model Rules to
reflect indigenous New York rules that had been incorporated over the years into its version of the Model Code .
</item 2>

<item 3>
American Bar Association Model Rules of Professional Conduct
In 2001 , the Commission for the Revision of the Rules of Professional Conduct of the State Bar of California began a comprehensive revision of the California
rules that would , among other things , convert them into a localized version of the ABA Model Rules . However , the Commission 's progress had been extremely
slow , simply because there are so many substantive and structural differences between the California rules and the ABA models . In October 2010 , the State Bar
of California formally adopted the Commission 's report , recommending adoption of 67 proposed changes to harmonise the California Rules of Professional Conduct
with the ABA Model Rules of Professional Responsibility . On September 19 , 2014 , however , for reasons that were not fully explained , the Supreme Court of
California rejected all of the proposed revisions and returned them to the State Bar . The Court 's letter directed the State Bar to start the process all over
again with a new Commission , and to submit a new set of revised rules by March 31 , 2017 . Professional regulation ( edit ) Except for the Patent Bar , there
is no national legal profession in the United States but there are 56 separate legal professions .
</item 3>

<item 4>
American Bar Association Model Rules of Professional Conduct
Jump up ^ http://www.mebaroverseers.org/attorney_regulation/maine_conduct_rules.html Jump up ^
http://www.maine.gov/tools/whatsnew/index.php?topic=mebar_overseers_bar_rules&id=88185&v=article Jump up ^ http://www.cnmilaw.org/pdf/court_rules/R07.pdf
Northern Mariana Islands Rules of Attorney Discipline and Procedure . ^ Jump up to : Rogers , Joan C. ( 2 October 2014 ) . `` California Justices Tell State Bar
to Redo Proposals for Updating Lawyer Conduct Rules '' . ABA / BNA Lawyers ' Manual on Professional Conduct . Bloomberg BNA . Retrieved 15 October 2014 . ^ Jump
up to : Zitrin , Richard ( 10 October 2014 ) . `` Viewpoint : State Supreme Court Resets Ethics Rewrite '' . The Recorder . ALM Media Properties , LLC .
Retrieved 15 October 2014 . Jump up ^ http://www.calbar.ca.gov/AboutUs/BoardofTrustees.aspx Board of Trustees , State Bar of California .
</item 4>

<item 5>
American Bar Association Model Rules of Professional Conduct
External links ( edit ) The ABA Model Rules of Professional Conduct Pre-2002 Changes to the Model Rules since its adoption in 1983 from Cornell Law School
Retrieved from `` https://en.wikipedia.org/w/index.php?title=American_Bar_Association_Model_Rules_of_Professional_Conduct&oldid=789593953 '' Categories : 1983
in law Legal ethics Codes of conduct American Bar Association United States law 1983 documents Hidden categories : Articles containing potentially dated
statements from 2015 All articles containing potentially dated statements Talk Contents About Wikipedia 한국어 Suomi Edit links This page was last edited on 8 July
2017 , at 08 : 30 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree
to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 5>

<item 6>
American Bar Association Model Rules of Professional Conduct
This is due to California law and public policy being so markedly different from the rest of the country in so many ways . For example , California is one of
only four states where covenants not to compete are void as against public policy ; and it is one of only three states where trusts are presumed to be revocable
unless the trust instrument explicitly states the trust is irrevocable ( whereas the opposite is true in the other 47 states ) . By way of comparative example ,
California has long held the position that clients must be encouraged to speak candidly to their legal counsel about both past and future actions knowing their
statements will be held in complete confidence . Their counsel can then explain the relevant law to them and urge them to conform their future conduct to the
law , thus promoting lawful conduct . In contrast , ABA Model Rule 1.6 contains exceptions to confidentiality for both violent and nonviolent ( e.g. financial )
future crimes , which California believes creates a strong disincentive for clients to discuss their future plans with their attorneys - but for which the
client might chose to comply with the law instead of violating it .
</item 6>

<item 7>
American Bar Association Model Rules of Professional Conduct
American Bar Association Model Rules of professional conduct - wikipedia American Bar Association Model Rules of professional conduct Jump to : navigation ,
search Professional responsibility Duties to the client Confidentiality Avoiding conflict of interest Due diligence and Competence ( law ) Avoid commingling
Avoid self - dealing Effective assistance Avoid fee splitting Withdrawal from representation Duties to the court Disclosure of perjury Disclosure of adverse
authority Duties to the profession Limitations on legal advertising Report misconduct Sources of law ABA Model Rules Penalties for misconduct Disbarment
Judicial misconduct The ABA Model Rules of Professional Conduct , created by the American Bar Association ( ABA ) , are a set of rules that prescribe baseline
standards of legal ethics and professional responsibility for lawyers in the United States . They were promulgated by the ABA House of Delegates upon the
recommendation of the Kutak Commission in 1983 . The rules are merely recommendations , or models , ( hence the name `` Model Rules '' ) and are not themselves
binding . However , having a common set of Model Rules facilitates a common discourse on legal ethics , and simplifies professional responsibility training as
well as the day - to - day application of such rules .
</item 7>

<item 8>
American Bar Association Model Rules of Professional Conduct
Even though New York did not adopt the Model Rules verbatim , the advantage of adopting its overall structure is that it simplifies the professional
responsibility training of New York lawyers , and makes it easier for out - of - state lawyers to conform their conduct to New York rules by simply comparing
their home state 's version of the Model Rules to New York 's version . In June 2009 , the Supreme Court of Maine approved the adoption of the Model Rules in
that state , effective August 1 , 2009 . Among the portions of the Model Rules that the Maine Rules of Professional Conduct do not include is Model Rule 1.8 ( j
) ( 2002 ) categorically prohibiting sexual relations between lawyer and client . In March 2015 , the Supreme Court of the Commonwealth of the Northern Mariana
Islands adopted its current Rules of Attorney Discipline and Procedure , which incorporates the ABA Model Rules by reference and expands upon them by adding
another 22 local rules of professional conduct . California ( edit ) California still has its own set of unique rules governing professional attorney conduct .
</item 8>

<item 9>
American Bar Association Model Rules of Professional Conduct
While either of their secretaries could be a member of the State Bar , they most likely are not . Similarly , the Board of Trustees that oversee the business
operations of the State Bar of California under supervision of the California Supreme Court is legally required to be a mix of attorneys licensed to practice in
California and `` public members '' who may not be licensed attorneys . So , while American attorneys are regulated by public sector agencies instead of private
sector law societies , most of the people who develop the regulatory policies are practising attorneys and sitting judges , so there is a degree of self -
regulation but not nearly to the same degree that solicitors and barristers in England and Wales , and other Commonwealth jurisdictions are self - regulated by
private sector associations . See also ( edit ) American Bar Association Model Code of Professional Responsibility American Bar Association References ( edit )
Jump up ^ Press Release : New Attorney Rules of Professional Conduct Announced , Communications Office of the New York Courts , 17 December 2008 .
</item 9>

<item 10>
American Bar Association Model Rules of Professional Conduct
California ( edit ) California still has its own set of unique rules governing professional attorney conduct . Although many of the rules are similar and
sometimes identical to the ABA Model Rules , the California rules differ markedly from the ABA models in both format , scope and content . For example , under
the ABA Model Rules , an attorney may be disciplined for the appearance of impropriety , regardless whether any actual impropriety occurred ; whereas the
California rules permit discipline for only actual impropriety . This reflects the nature of California law . It is unlikely that California will ever adopt the
ABA Model Rules apart beyond harmonising the format and existing California rules that are identical or nearly identical to ABA Model Rules . Reasons for this
reluctance include the fact there are more than 280,000 attorneys licensed in California , so a drastic change from the existing rules to completely different
rules would shock the profession and , ultimately , would most likely fail . Additionally , many of California 's rules of professional conduct are established
by statute and , thus , would require action by the California Legislature to change them ; or , alternatively , action by the people of California through the
voter initiative process .
</item 10>
=== END ITEM id=0e180567056a5dd4 ===
