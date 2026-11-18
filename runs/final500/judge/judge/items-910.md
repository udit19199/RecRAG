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

=== ITEM id=df3f5a89865b6d4c ===
Gold sentences and facts:
Sentence 0: The Walking Dead Cover of The Walking Dead No. 1 . Art by Tony Moore . Publication information Publisher Image Comics Schedule Monthly Format Ongoing series Genre Post-apocalyptic Publication date October 2003 -- present No. of issues 178 Creative team Created by Robert Kirkman Tony Moore Written by Robert Kirkman Artist ( s ) Tony Moore ( # 1 -- 6 ) Charlie Adlard ( # 7 -- present ) Penciller ( s ) Tony Moore ( # 1 -- 6 ) Charlie Adlard ( # 7 -- present ) Inker ( s ) Tony Moore ( # 1 -- 6 ) Charlie Adlard ( # 7 -- present ) Stefano Gaudiano ( # 115 -- present ) Letterer ( s ) Robert Kirkman ( # 1 -- 19 ) Rus Wooton ( # 20 -- present ) Colorist ( s ) Tony Moore ( # 1 -- 5 ) Cliff Rathburn ( # 6 -- present ) Editor ( s ) Aubrey Sitterson ( # 55 -- 70 ) Sina Grace ( # 67 -- 96 ) Sean Mackiewicz ( # 97 -- present )
  Fact 0.0: The Walking Dead No. 1 has a cover with art by Tony Moore.
  Fact 0.1: The Walking Dead is published by Image Comics.
  Fact 0.2: The Walking Dead has a monthly publication schedule.
  Fact 0.3: The Walking Dead is an ongoing series.
  Fact 0.4: The Walking Dead is a post-apocalyptic comic.
  Fact 0.5: The Walking Dead was published from October 2003 to the present.
  Fact 0.6: The Walking Dead has 178 issues.
  Fact 0.7: The Walking Dead was created by Robert Kirkman and Tony Moore.
  Fact 0.8: Robert Kirkman wrote The Walking Dead.
  Fact 0.9: Tony Moore was the artist for issues 1 to 6 of The Walking Dead.
  Fact 0.10: Charlie Adlard was the artist from issue 7 of The Walking Dead to the present.
  Fact 0.11: Tony Moore was the penciller for issues 1 to 6 of The Walking Dead.
  Fact 0.12: Charlie Adlard was the penciller from issue 7 of The Walking Dead to the present.
  Fact 0.13: Tony Moore was the inker for issues 1 to 6 of The Walking Dead.
  Fact 0.14: Charlie Adlard was the inker from issue 7 of The Walking Dead to the present.
  Fact 0.15: Stefano Gaudiano was an inker of The Walking Dead from issue 115 to the present.
  Fact 0.16: Robert Kirkman was the letterer of The Walking Dead for issues 1 to 19.
  Fact 0.17: Rus Wooton was the letterer of The Walking Dead from issue 20 to the present.
  Fact 0.18: Tony Moore was the colorist of The Walking Dead for issues 1 to 5.
  Fact 0.19: Cliff Rathburn was the colorist of The Walking Dead from issue 6 to the present.
  Fact 0.20: Aubrey Sitterson was an editor of The Walking Dead for issues 55 to 70.
  Fact 0.21: Sina Grace was an editor of The Walking Dead for issues 67 to 96.
  Fact 0.22: Sean Mackiewicz was an editor of The Walking Dead from issue 97 to the present.
Sentence 1: The Walking Dead debuted in 2003 , published by Image Comics , with art by Tony Moore for the first six issues and Cliff Rathburn shading the art after issue five . Charlie Adlard took over as artist on issue # 7 , after he was approached by Kirkman . Moore also did the cover art for the first twenty - four issues and the first four trade paperbacks . The remainder were done by Adlard .
  Fact 1.0: The Walking Dead debuted in 2003.
  Fact 1.1: The Walking Dead was published by Image Comics.
  Fact 1.2: Tony Moore did the art for the first six issues of The Walking Dead.
  Fact 1.3: Cliff Rathburn shaded the art of The Walking Dead after issue five.
  Fact 1.4: Charlie Adlard took over as artist of The Walking Dead on issue 7.
  Fact 1.5: Robert Kirkman approached Charlie Adlard before Adlard took over as artist on issue 7.
  Fact 1.6: Tony Moore did the cover art for the first twenty-four issues of The Walking Dead.
  Fact 1.7: Tony Moore did the cover art for the first four trade paperbacks of The Walking Dead.
  Fact 1.8: Charlie Adlard did the remainder of the cover art of The Walking Dead.

Retrieved items:
<item 1>
The Walking Dead (comic book)
25 : No Turning Back 1 - 63215 - 659 - 8 April 5 , 2016 The Walking Dead # 145 -- 150 August 2015 -- January 2016 13 7 The Walking Dead Vol. 26 : Call To Arms 1
- 63215 - 917 - 1 September 14 , 2016 The Walking Dead # 151 -- 156 February 2016 -- July 2016 The Walking Dead Vol. 27 : The Whisperer War 1 - 53430 - 052 - X
March 7 , 2017 The Walking Dead # 157 -- 162 August 2016 -- January 2017 14 The Walking Dead Vol. 28 : A Certain Doom 1 - 53430 - 244 - 1 October 3 , 2017 The
Walking Dead # 163 -- 168 February 2017 -- June 2017 The Walking Dead Vol.
</item 1>

<item 2>
The Walking Dead (comic book)
21 : All Out War ( Part Two ) 1 - 63215 - 030 - 1 July 29 , 2014 The Walking Dead # 121 -- 126 February 2014 -- April 2014 11 6 The Walking Dead Vol. 22 : A New
Beginning 1 - 63215 - 041 - 7 November 5 , 2014 The Walking Dead # 127 -- 132 May 2014 -- October 2014 The Walking Dead Vol. 23 : Whispers Into Screams 1 -
63215 - 258 - 4 May 12 , 2015 The Walking Dead # 133 -- 138 October 2014 -- March 2015 12 The Walking Dead Vol. 24 : Life And Death 1 - 63215 - 402 - 1 August
26 , 2015 The Walking Dead # 139 -- 144 March 2015 -- July 2015 The Walking Dead Vol.
</item 2>

<item 3>
The Walking Dead (comic book)
17 : Something To Fear 1 - 60706 - 615 - 7 November 21 , 2012 The Walking Dead # 97 -- 102 May 2012 -- September 2012 9 5 The Walking Dead Vol. 18 : What Comes
After 1 - 60706 - 687 - 4 June 18 , 2013 The Walking Dead # 103 -- 108 October 2012 -- March 2013 The Walking Dead Vol. 19 : March To War 1 - 60706 - 818 - 4
November 13 , 2013 The Walking Dead # 109 -- 114 April 2013 -- September 2013 10 The Walking Dead Vol. 20 : All Out War ( Part One ) 1 - 60706 - 882 - 6 March
11 , 2014 The Walking Dead # 115 -- 120 October 2013 -- January 2014 The Walking Dead Vol.
</item 3>

<item 4>
The Walking Dead (comic book)
Contents ( hide ) 1 Publication history 2 Story arcs 3 Specials 4 List of characters 4.1 The undead / reanimated 5 Reception 6 Other media 6.1 Television
adaptations 6.2 Animation comic 6.3 The Walking Dead : The Game 6.4 The Walking Dead : All Out War 6.5 Novels 7 Collected editions 7.1 Trade paperbacks 7.2
Hardcovers 7.3 Omnibus editions 7.4 Compendium editions 7.5 Other publications 8 References 8.1 Comic books 9 External links Publication history ( edit ) The
original pitch by Kirkman and Moore was for a followup to George A. Romero 's Night of the Living Dead , with the series taking place in the 1960s . Image
Comics co-founder Jim Valentino suggested using an original concept instead so the creators would own the property outright . The revised pitch became The
Walking Dead . The Walking Dead debuted in 2003 , published by Image Comics , with art by Tony Moore for the first six issues and Cliff Rathburn shading the art
after issue five .
</item 4>

<item 5>
The Walking Dead (comic book)
2005 Collects # 1 -- 24 The Walking Dead : Volume 2 Deluxe HC 1 - 60706 - 029 - 9 February 17 , 2009 Collects # 25 -- 48 The Walking Dead : Volume 3 Deluxe HC 1
- 60706 - 330 - 1 February 2 , 2011 Collects # 49 -- 72 The Walking Dead : Volume 4 Deluxe HC 1 - 60706 - 616 - 5 December 12 , 2012 Collects # 73 -- 96 The
Walking Dead : Volume 5 Deluxe HC 1 - 63215 - 042 - 5 November 19 , 2014 Collects # 97 -- 120 The Walking Dead : Volume 6 Deluxe HC 1 - 63215 - 521 - 4 December
2 , 2015 Collects # 121 -- 144 The Walking Dead : Volume 7 Deluxe HC 1 - 53430 - 335 - 9 November 7 , 2017 Collects # 145 -- 168 Compendium editions ( edit )
Softcover compendium editions collect 48 comic issues each .
</item 5>

<item 6>
The Walking Dead (comic book)
Charlie Adlard took over as artist on issue # 7 , after he was approached by Kirkman . Moore also did the cover art for the first twenty - four issues and the
first four trade paperbacks . The remainder were done by Adlard . When the television series premiered in October 2010 , Image Comics announced The Walking Dead
Weekly . The first 52 issues of the series began to be reprinted on January 5 , 2011 , with one issue per week for a year . The series is periodically
re-published in trade paperbacks which contain six issues each , hardcover books with twelve issues and occasional bonus material , omnibus editions of twenty -
four issues , and compendium editions of forty - eight issues . Story arcs ( edit ) Volume 1 : Days Gone Bye ( Issues 1 -- 6 ) Rick Grimes , a sheriff 's deputy
from Kentucky , is wounded in the line of duty and emerges from a coma to find the world overrun by the undead . He returns home to find his house ransacked and
his wife and son gone .
</item 6>

<item 7>
The Walking Dead (comic book)
12 : Life Among Them 1 - 60706 - 254 - 2 August 3 , 2010 The Walking Dead # 67 -- 72 November 2009 -- May 2010 The Walking Dead Vol. 13 : Too Far Gone 1 - 60706
- 329 - 8 November 23 , 2010 The Walking Dead # 73 -- 78 June 2010 -- October 2010 7 The Walking Dead Vol. 14 : No Way Out 1 - 60706 - 392 - 1 June 22 , 2011
The Walking Dead # 79 -- 84 November 2010 -- April 2011 The Walking Dead Vol. 15 : We Find Ourselves 1 - 60706 - 440 - 5 December 27 , 2011 The Walking Dead #
85 -- 90 May 2011 -- October 2011 8 The Walking Dead Vol. 16 : A Larger World 1 - 60706 - 559 - 2 June 6 , 2012 The Walking Dead # 91 -- 96 November 2011 --
April 2012 The Walking Dead Vol.
</item 7>

<item 8>
The Walking Dead (comic book)
The Walking Dead ( comic book ) - Wikipedia The Walking Dead ( comic book ) Jump to : navigation , search The Walking Dead Cover of The Walking Dead No. 1 . Art
by Tony Moore . Publication information Publisher Image Comics Schedule Monthly Format Ongoing series Genre Post-apocalyptic Publication date October 2003 --
present No.
</item 8>

<item 9>
The Walking Dead (comic book)
29 : Lines We Cross 1 - 53430 - 497 - 5 March 13 , 2018 The Walking Dead # 169 -- 174 July 2017 -- December 2017 15 8 Hardcovers ( edit ) All hardcovers contain
the contents of the comics , including the covers , and in some cases bonus material . The books ' trim size is larger than the paperbacks . Each hardcover
contains two story arcs from the series . Signed versions of the books are available , each limited to 310 pieces . Title ISBN Release Date Collected Material
Cover Character The Walking Dead : Book One 978 - 1582406190 July 19 , 2006 The Walking Dead # 1 -- 12 Rick The Walking Dead : Book Two 978 - 1582406985 March 7
, 2007 The Walking Dead # 13 -- 24 Michonne The Walking Dead : Book Three 978 - 1582408255 December 19 , 2007 The Walking Dead # 25 -- 36 The Governor The
Walking Dead : Book Four 978 - 1607060000 October 29 ,
</item 9>

<item 10>
The Walking Dead (comic book)
2008 The Walking Dead # 37 -- 48 Lori and Judith The Walking Dead : Book Five 978 - 1607061717 May 5 , 2010 The Walking Dead # 49 -- 60 Abraham The Walking Dead
: Book Six 978 - 1607063278 October 26 , 2010 The Walking Dead # 61 -- 72 Carl The Walking Dead : Book Seven 978 - 1607064398 October 18 , 2011 The Walking Dead
# 73 -- 84 Rick The Walking Dead : Book Eight 978 - 1607065937 September 26 , 2012 The Walking Dead # 85 -- 96 Andrea The Walking Dead : Book Nine 978 -
1607067986 September 17 , 2013 The Walking Dead # 97 -- 108 Negan The Walking Dead : Book Ten 978 - 1632150349 September 3 , 2014 The Walking Dead # 109 -- 120
Ezekiel The Walking Dead : Book Eleven 978 - 1632152718 February 25 ,
</item 10>
=== END ITEM id=df3f5a89865b6d4c ===

=== ITEM id=1e4cc9db7ce87885 ===
Gold sentences and facts:
Sentence 0: The earliest predecessors of the eye were photoreceptor proteins that sense light , found even in unicellular organisms , called `` eyespots '' . Eyespots can only sense ambient brightness : they can distinguish light from dark , sufficient for photoperiodism and daily synchronization of circadian rhythms . They are insufficient for vision , as they can not distinguish shapes or determine the direction light is coming from . Eyespots are found in nearly all major animal groups , and are common among unicellular organisms , including euglena . The euglena 's eyespot , called a stigma , is located at its anterior end . It is a small splotch of red pigment which shades a collection of light sensitive crystals . Together with the leading flagellum , the eyespot allows the organism to move in response to light , often toward the light to assist in photosynthesis , and to predict day and night , the primary function of circadian rhythms . Visual pigments are located in the brains of more complex organisms , and are thought to have a role in synchronising spawning with lunar cycles . By detecting the subtle changes in night - time illumination , organisms could synchronise the release of sperm and eggs to maximise the probability of fertilisation .
  Fact 0.0: The earliest predecessors of the eye were photoreceptor proteins that sense light.
  Fact 0.1: The light-sensing photoreceptor proteins that were the earliest predecessors of the eye are found even in unicellular organisms.
  Fact 0.2: These early light-sensing structures are called 'eyespots'.
  Fact 0.3: Eyespots can only sense ambient brightness.
  Fact 0.4: Eyespots can distinguish light from dark.
  Fact 0.5: Sensing ambient brightness is sufficient for photoperiodism.
  Fact 0.6: Sensing ambient brightness is sufficient for daily synchronization of circadian rhythms.
  Fact 0.7: Eyespots are insufficient for vision.
  Fact 0.8: Eyespots cannot distinguish shapes.
  Fact 0.9: Eyespots cannot determine the direction light is coming from.
  Fact 0.10: Eyespots are found in nearly all major animal groups.
  Fact 0.11: Eyespots are common among unicellular organisms.
  Fact 0.12: Euglena is a unicellular organism with an eyespot.
  Fact 0.13: The euglena's eyespot is called a stigma.
  Fact 0.14: The euglena's stigma is located at its anterior end.
  Fact 0.15: The euglena's stigma is a small splotch of red pigment.
  Fact 0.16: The euglena's stigma shades a collection of light-sensitive crystals.
  Fact 0.17: Together with the leading flagellum, the euglena's eyespot allows the organism to move in response to light.
  Fact 0.18: The euglena often moves toward the light to assist in photosynthesis.
  Fact 0.19: The euglena's eyespot allows the organism to predict day and night.
  Fact 0.20: Predicting day and night is the primary function of circadian rhythms.
  Fact 0.21: Visual pigments are located in the brains of more complex organisms.
  Fact 0.22: Visual pigments are thought to have a role in synchronising spawning with lunar cycles.
  Fact 0.23: Organisms could detect subtle changes in night-time illumination.
  Fact 0.24: By detecting night-time illumination changes, organisms could synchronise the release of sperm and eggs.
  Fact 0.25: Synchronising the release of sperm and eggs maximises the probability of fertilisation.

Retrieved items:
<item 1>
Evolution of the eye
This suggests that when the two lineages diverged in the Precambrian , they had only very primitive light receptors , which developed into more complex eyes
independently . Early eyes ( edit ) The basic light - processing unit of eyes is the photoreceptor cell , a specialized cell containing two types of molecules
in a membrane : the opsin , a light - sensitive protein , surrounding the chromophore , a pigment that distinguishes colors . Groups of such cells are termed ``
eyespots '' , and have evolved independently somewhere between 40 and 65 times . These eyespots permit animals to gain only a very basic sense of the direction
and intensity of light , but not enough to discriminate an object from its surroundings . Developing an optical system that can discriminate the direction of
light to within a few degrees is apparently much more difficult , and only six of the thirty - some phyla possess such a system . However , these phyla account
for 96 % of living species . The planarian has `` cup '' eyespots that can slightly distinguish light direction .
</item 1>

<item 2>
Evolution of the eye
In both vertebrates and octopodes , 1 represents the retina , 2 represents the nerve fibers , and 3 represents the optic nerve . Main article : Evolutionary
baggage The eyes of many animals record their evolutionary history in their contemporary anatomy . The vertebrate eye , for instance , is built `` backwards and
upside down '' , requiring `` photons of light to travel through the cornea , lens , aqueous fluid , blood vessels , ganglion cells , amacrine cells ,
horizontal cells , and bipolar cells before they reach the light - sensitive rods and cones that transduce the light signal into neural impulses , which are
then sent to the visual cortex at the back of the brain for processing into meaningful patterns . '' While such a construct has some drawbacks , it also allows
the outer retina of the vertebrates to sustain higher metabolic activities as compared to the non-inverted design . It also allowed for the evolution of the
choroid layer , including the retinal pigment epithelial ( RPE ) cells , which play an important role in protecting the photoreceptive cells from photo -
oxidative damage .
</item 2>

<item 3>
The references discuss eye evolution from several angles: exceptionally preserved Early Cambrian arthropod eyes show sophisticated optics; fossil evidence
traces early vertebrate eye development; and research examines eye development, photoreceptors, optical origins, and the functional basis of vision. Nilsson and
Pelger estimated the time needed for an eye to evolve, while other works explore shared ancestral genes and the role of vision in evolutionary change. Sources
include scientific papers by Lee et al., Gehring, Nilsson, Young, and
</item 3>

<item 4>
Evolution of the eye
Nilsson and S. Pelger estimated in a classical paper how many generations are needed to evolve a complex eye in vertebrates . Another researcher , G.C. Young ,
has used the fossil record to infer evolutionary conclusions , based on the structure of eye orbits and openings in fossilized skulls for blood vessels and
nerves to go through . All this adds to the growing amount of evidence that supports Darwin 's theory . Rate of evolution ( edit ) The first fossils of eyes
found to date are from the lower Cambrian period ( about 540 million years ago ) . The lower Cambrian had a burst of apparently rapid evolution , called the ``
Cambrian explosion '' . One of the many hypotheses for `` causes '' of the Cambrian explosion is the `` Light Switch '' theory of Andrew Parker : It holds that
the evolution of eyes started an arms race that accelerated evolution . Before the Cambrian explosion , animals may have sensed light , but did not use it for
fast locomotion or navigation by vision . The rate of eye evolution is difficult to estimate , because the fossil record , particularly of the lower Cambrian ,
is poor .
</item 4>

<item 5>
Evolution of the eye
How fast a circular patch of photoreceptor cells evolve into a fully functional vertebrate eye has been estimated based on rates of mutation , relative
advantage to the organism , and natural selection . However , the time needed for each state was consistently overestimated and the generation time was set to
one year , which is common in small animals . Even with these pessimistic values , the vertebrate eye would still evolve from a patch of photoreceptor cells in
less than 364,000 years . One origin or many ? ( edit ) Whether the eye evolved once or many times depends on the definition of an eye . All eyed animals share
much of the genetic machinery for eye development . This suggests that the ancestor of eyed animals had some form of light - sensitive machinery -- even if it
was not a dedicated optical organ . However , even photoreceptor cells may have evolved more than once from molecularly similar chemoreceptor cells . Probably ,
photoreceptor cells existed long before the Cambrian explosion .
</item 5>

<item 6>
Evolution of the eye
1 ( 4 ) : 439 -- 447 . doi : 10.1007 / s12052 - 008 - 0084 - 1 . ISSN 1936 - 6426 . Further reading ( edit ) Lamb TD , Collin SP , Pugh EN ( December 2007 ) .
`` Evolution of the vertebrate eye : opsins , photoreceptors , retina and eye cup '' . Nat. Rev. Neurosci. 8 ( 12 ) : 960 -- 76 . doi : 10.1038 / nrn2283 . PMC
3143066 . PMID 18026166 . Illustration . Review Lamb , TD ( 2011 ) . `` Evolution of the Eye '' ( PDF ) . Scientific American . 305 : 64 -- 69 . doi : 10.1038 /
scientificamerican0711 - 64 . Land , Michael F. ; Nilsson , Dan - Eric ( 2012 ) . `` The origin of vision '' . Animal Eyes ( 2 ed . ) . Oxford : Oxford
University Press .
</item 6>

<item 7>
Evolution of the eye
However , he went on that despite the difficulty in imagining it , its evolution was perfectly feasible : ... if numerous gradations from a simple and imperfect
eye to one complex and perfect can be shown to exist , each grade being useful to its possessor , as is certainly the case ; if further , the eye ever varies
and the variations be inherited , as is likewise certainly the case and if such variations should be useful to any animal under changing conditions of life ,
then the difficulty of believing that a perfect and complex eye could be formed by natural selection , though insuperable by our imagination , should not be
considered as subversive of the theory . He suggested a stepwise evolution from `` an optic nerve merely coated with pigment , and without any other mechanism
'' to `` a moderately high stage of perfection '' , and gave examples of existing intermediate steps . Darwin 's suggestions were soon shown to be correct , and
current research is investigating the genetic mechanisms underlying eye development and evolution . Biologist D.E. Nilsson has independently theorized about
four general stages in the evolution of a vertebrate eye from a patch of photoreceptors .
</item 7>

<item 8>
Evolution of the eye
149 Jump up ^ Lee , M.S.Y. ; Jago , J.B. ; Garcia - Bellido , D.C. ; Edgecombe , G.E. ; Gehling , J. G ; Paterson , J.R. ( 2011 ) . `` Modern optics in
exceptionally preserved eyes of Early Cambrian arthropods from Australia '' . Nature . 474 : 631 -- 634 . doi : 10.1038 / nature10097 . PMID 21720369 . ^ Jump
up to : Darwin , Charles ( 1859 ) . On the Origin of Species . London : John Murray . Jump up ^ Gehring WJ ( 2005 ) . `` New perspectives on eye development and
the evolution of eyes and photoreceptors '' . J. Hered. 96 ( 3 ) : 171 -- 84 . doi : 10.1093 / jhered / esi027 . PMID 15653558 . Jump up ^ Nilsson , D. - E. (
2013 ) . `` Eye evolution and its functional basis '' . Visual Neuroscience.
</item 8>

<item 9>
Evolution of the eye
Evolution of the eye - wikipedia Evolution of the eye Jump to : navigation , search Major stages in the evolution of the eye in vertebrates . The evolution of
the eye is attractive to study , because the eye distinctively exemplifies an analogous organ found in many animal forms . Complex , image - forming eyes have
evolved independently several times . The evolutionary geneticist Walter Gering holds a different opinion . According his theory , the genes engaged with
photoreceptor cells ( so called Pax 6 genes ) appeared probably at once in the life history , at a very early stage of the evolution . Probably , the complex ,
image forming kinds of eyes are at least five : Chidaria - lens eye , complex eye ; Molluscs - lens eye , everse retina ; Annelids - lens eye , everse retina ;
Arthropods - facet eye ; Vertebrate - lens eye , inverse retina Complex eyes appeared first within the few million years of the Cambrian explosion . From before
the Cambrian , no evidence of eyes has survived , but diverse eyes are known from the Burgess shale of the Middle Cambrian , and from the slightly older Emu Bay
Shale .
</item 9>

<item 10>
Evolution of the eye
`` Eye evolution and its functional basis '' . Visual Neuroscience. 30 : 5 -- 20 . doi : 10.1017 / s0952523813000035 . ^ Jump up to : Nilsson , D. - E. ; Pelger
, S. ( 1994 ) . `` A pessimistic estimate of the time required for an eye to evolve '' . Proceedings of the Royal Society B : Biological Sciences . 256 : 53 --
58 . doi : 10.1098 / rspb. 1994.0048 . PMID 8008757 . Jump up ^ Young , G.C. ( 2008 ) . `` Early evolution of the vertebrate eye -- fossil evidence '' . Evo Edu
Outreach. 1 : 427 -- 438 . doi : 10.1007 / s12052 - 008 - 0087 - y . Jump up ^ Parker , A.R. ( 2009 ) . `` On the origin of optics '' . Optics & Laser
Technology . 43 ( 2 ) : 323 -- 329 .
</item 10>
=== END ITEM id=1e4cc9db7ce87885 ===

=== ITEM id=eace0b33d72b37ae ===
Gold sentences and facts:
Sentence 0: No . Official English title Original Japanese title Original air date English air date 151 `` Sabertooth '' `` Seibātūsu '' ( 剣 咬 の 虎 ( セイバー トゥース ) ) October 6 , 2012 November 17 , 2015 The Legion Platoon is tasked by Zentopia 's archbishop to retrieve the missing pieces of the Infinity Clock and seal them again . Before embarking on their mission , they visit Fairy Tail and make amends with the guild for their previous hostility against them . It is later discovered that Fairy Tail has lost their title as the strongest guild in Fiore to the Saber Tooth guild over the past seven years . Two of the strongest wizards in the guild include a pair of Dragon Slayers named Sting and Rogue , who are obsessed with fighting Natsu and Gajeel , respectively . Meanwhile , Makarov brings Gildarts to their abandoned guildhall and shows him Lumen Histoire , a light that Makarov says is Fairy Tail 's greatest secret . Makarov then tells Gildarts that he has been chosen to become the next master of Fairy Tail . 152 `` And So We Aim for the Top '' `` Soshite Ore - tachi wa Chōjō o Mezasu '' ( そして オレ たち は 顶 上 を 目指す ) October 13 , 2012 November 24 , 2015 Natsu and his friends discover that a power gap has developed between them and the rest of the guild members who were not on Tenrou Island . The group meets with Poluchka for medicine to strengthen their magic , but are unsuccessful . Wendy recognizes Poluchka 's voice and scent as those of Grandine , the dragon who raised her . Poluchka reveals herself as the Edolas counterpart of Grandine , and gives Wendy written instructions for two advanced spells that Grandine failed to teach her . Meanwhile , Makarov announces Gildarts ' succession as master to the rest of the guild , only to find he has gone on a journey , leaving a letter of resignation behind . In the letter , Gildarts reinstates Laxus into the guild , returns mastership to Makarov , and tells him to make Fairy Tail the strongest guild in Fiore again . Romeo suggests they participate in the Grand Magic Games , an annual festival designed to determine the guild rankings in Fiore . 153 `` Song of the Stars '' `` Hoshiboshi no Uta '' ( 星々 の 歌 ) October 20 , 2012 December 1 , 2015 With the Grand Magic Games occurring in three months time , Natsu and his friends train at a beach resort . On their second day of training , Virgo appears and tells the wizards that the Celestial Spirit World is in danger of destruction . When the wizards arrive , they instead find a surprise party held by the Celestial Spirit King celebrating their return from Sirius Island after seven years . They spend the day in the spirit world partying with the spirits . Before they leave , however , Virgo informs them that one day in the spirit world equals three months in the human world , and they find the Grand Magic Games are upon them . 154 `` For All the Time We Missed Each Other '' `` Surechigatta Jikan no Bun dake '' ( すれ 违 っ た 时间 の 分 だけ ) October 27 , 2012 December 8 , 2015 While mulling over their lost training time , the Fairy Tail wizards receive a message via carrier pigeon requesting a private meeting in the forest nearby . There they meet a runaway Jellal , Ultear , and Merudy , who have formed their own guild , Crime Sorcière , to defeat Zeref and any dark guilds that try to use his power . The three ask Fairy Tail for their help in determining the source of a magical anomaly similar to Zeref 's magic that they have sensed around the Grand Magic Games arena , Domus Flau . In exchange , Ultear uses her magic to invoke their Second Origin in order to double their magic power . As the others undergo the painful process of awakening their Second Origin , Erza talks with Jellal over his regained memories . The two nearly kiss , but Jellal decides that he and Erza can not be together and lies to her that he has a fiancée . Erza quickly picks up on this and respects his decision , seeing Crime Sorcière off as they leave . 155 `` Crocus , the Flower - Blooming Capital '' `` Hanasaku Miyako : Kurokkasu '' ( 花 笑 く 都 ・ クロッカス ) November 3 , 2012 December 15 , 2015 Natsu , Lucy , Gray , Erza , and Wendy are chosen to represent their guild in the Grand Magic Games , which are to be held in Fiore 's capital , Crocus . As they tour the city , Natsu and Happy encounter the twin dragon slayers , Sting and Rogue , who mock Natsu for not being able to kill Acnologia , and boast of becoming true dragon slayers by killing the dragons who raised them . At midnight , Fairy Tail and the other participating guilds are put through a preliminary event -- a race through a massive floating labyrinth -- designed to narrow the number of participating guilds down to eight teams . 156 `` Sky Labyrinth '' `` Sukai Rabirinsu '' ( 空中 迷宮 ( スカイ ラビリンス ) ) November 10 , 2012 December 22 , 2015 Natsu 's team notices that Wendy has gone missing , so Elfman replaces her to keep them from being eliminated . As they make their way through the enchanted labyrinth , the five encounter various other guilds and steal the maps made by the other guilds to better navigate the maze . They manage to reach the end of the maze and pass the elimination round , confident they have arrived first , only to discover they have barely taken last place . Meanwhile , Happy , Lisanna , and the rest of the guild search for Wendy and Carla . They eventually find the two in the garden of the king 's palace , Mercurius , with their magic power depleted . 157 `` New Guild '' `` Shinki Girudo '' ( 新 规 ギルド ) November 17 , 2012 December 29 , 2015 Wendy 's body is weakened as a result of losing her magic power . As Poluchka helps her recover , she formally requests Elfman to take her place in the Grand Magic Games . Fairy Tail steps into the arena with a negative audience reception , but they are bolstered by their guild mates and the spirit of Mavis , whom only the Fairy Tail guild members can see . The next four participating teams are revealed to be Quatro Cerberus , Mermaid Heel , Blue Pegasus , and Lamia Scale . Following them is Raven Tail , a former dark guild led by Makarov 's son Ivan , who confess to attacking Wendy and Carla . To everyone 's shock , the next team is revealed to be a second team from Fairy Tail composed of Gajeel , Juvia , Mirajane , Laxus , and Jellal ( disguised as Mystogan to help investigate the arena ) . The final guild to be announced is Saber Tooth , much to the joy of the audience . With all eight teams announced , they each send out one member to participate in the first contest of the games , with Gray and Juvia representing their teams . 158 `` Night of Shooting Stars '' `` Hoshi Furu Yoru ni '' ( 星 降 ル 夜 ニ ) November 24 , 2012 January 5 , 2016 The games ' first contest , `` Hidden '' , requires the eight participants to score points by finding and attacking each other in a virtual city filled with copies of themselves . Raven Tail member Nalpudding repeatedly targets Gray , who scores last place in the contest while Saber Tooth 's Rufus scores first by using his Memory - Make magic to attack all the other participants at once . Humiliated , Gray swears revenge against Raven Tail and Saber Tooth . Afterwards , Lucy is chosen to fight Raven Tail member Flare in the tournament 's first battle round . 159 `` Lucy vs. Flare '' `` Rūshii vs. Furea '' ( ルーシィ vs. フレア ) December 1 , 2012 January 12 , 2016 Lucy displays tremendous skill in her battle against Flare by being able to summon two spirits simultaneously . As Lucy gains the upper hand , Flare cheats by using her magic hair to hold Alzack and Bisca 's daughter Asuka as a hostage , blackmailing Lucy into not fighting back to keep her from harming Asuka . As Lucy tearfully speaks of Asuka 's danger , Natsu is able to hear it and rushes to Asuka to incinerate Flare 's hair , allowing Lucy to counterattack with her Urano Metoria spell in combination with Gemini . However , before the spell fully activates , Flare 's teammate Obra nullifies Lucy 's magic , causing her to collapse and lose the match . 160 `` Portent '' `` Kyōzui '' ( 凶 瑞 ) December 8 , 2012 January 19 , 2016 Blue Pegasus ' Ren defeats Mermaid Heel 's Araña during the games ' second match , while Saber Tooth 's Orga effortlessly defeats Quatro Cerberus ' War Cry in the third match . For the day 's fourth and final match , Jellal is pitted against Lamia Scale 's Jura , intending to avenge the humiliated Fairy Tail members . He mimics Mystogan 's magic to keep up his disguise , but he finds it ineffective to use against Jura and is forced to use his own magic instead . Before he can cast a spell that will expose his true identity , Ultear uses Merudy 's sensory to force Jellal into submission . Meanwhile , the hospitalized Carla awakens after having a troubling premonition of the future . 161 `` Chariots '' `` Chariotto '' ( 战 车 ( チャリオット ) ) December 15 , 2012 January 26 , 2016 Fairy Tail is undaunted by their poor performance on the first day . As the guild celebrates , they are visited and taunted by Bacchus , Erza 's rival from Quatro Cerberus , who has come to replace War Cry . Carla relays her premonition to Poluchka , saying that she saw Mercurius collapsing around what appeared to be a tearfully singing Lucy . The second day of the tournament begins with a footrace atop a train of moving chariots , causing the participants Natsu , Gajeel , and Sting to suffer from motion sickness as a side effect of their Dragon Slayer magic . As Natsu and Gajeel force themselves forward , Sting inquires about their determination to which Natsu responds that it is as a result of the strength of Fairy Tail 's bonds . Sting eventually throws the race while Natsu and Gajeel manage to finish as sixth and seventh placers , respectively , earning Fairy Tail the audience 's respect . 162 `` Elfman vs. Bacchus '' `` Erufuman vs. Bakkasu '' ( エルフマン vs. バッカス ) December 22 , 2012 September 1 , 2015 Due to a miscomunnication between the king and his knight Arcadios , Elfman is chosen as Bacchus ' opponent instead of Erza . Before the match commences , the two opponents make a wager : if Bacchus wins , he will court Elfman 's sisters , and if Elfman wins , Bacchus ' team name will be changed to `` Quatro Puppy '' for the rest of the games . Bacchus ruthlessly attacks Elfman with his bare palms , who manages to damage Bacchus with his defensive Lizardman transformation . Elfman manages to outlast Bacchus as he exhausts himself , winning the match and forcing Bacchus ' team to be renamed . Meanwhile , Natsu finds Wendy , Carla , and Poluchka being kidnapped by a group of masked men . Upon being caught , the men claim they were hired to kidnap Lucy by Raven Tail . They are soon revealed to have actually been sent by Arcadios , who intends to sow discord between the guilds as he plots to use Lucy for his `` Eclipse Project '' . 163 `` Mirajane vs. Jenny '' `` Mirajēn vs. Jenī '' ( ミラ ジェーン vs. ジェニー ) January 5 , 2013 September 1 , 2015 The second day of the Grand Magic Games continues , with Mirajane of Fairy Tail Team B fighting against Jenny Realight of Team Blue Pegasus . Since the two competitors are well - known for being magazine models , their battle consists almost entirely of a fashion showdown , much to the pleasure of the audience . Eventually as the showdown becomes redundant , Jenny proposes a wager : the loser must pose nude for the Sorcerer Magazine , which Mirajane accepts . Jenny then transforms into her battle armor while Mirajane shifts into her Satan Soul `` Sitri '' form and easily defeats Jenny . The final match of the day is announced , with Kagura Mikazuchi of Mermaid Heel pitted against Yukino Aguria of Saber Tooth . Yukino proposes to Kagura that they wager their lives on their battle . Meanwhile , Arcadios is confronted by Darton , the kingdom 's defense minister , about his `` Eclipse Project '' . 164 `` Kagura vs. Yukino '' `` Kagura vs. Yukino '' ( カグラ vs. ユキノ ) January 12 , 2013 September 1 , 2015 Yukino is revealed to be a celestial spirit wizard when she summons Libra and Pisces to attack Kagura . Using her gravity - altering magic , Kagura defeats the two spirits , thus forcing Yukino to use a Gatekey that summons Ophiuchus , which is regarded as the strongest among the Celestial Spirits . However , Kagura slays Ophiuchus with her sword still sheathed and easily defeats Yukino . Later , Jellal discusses the fact that he has not sensed the evil power similar to Zeref 's even though the second day of the Grand Magic Games has ended . Erza reunites with Milliana , who is revealed to be a member of Mermaid Heel . Meanwhile , Saber Tooth 's master Jiemma is dissatisfied with his guild 's performance . Though Sting is given another chance , Jiemma humiliates Yukino for her failure . 165 `` Hatred at Nightfall '' `` Urami wa Yoru no Tobari ni Tsutsumarete '' ( 怨み は 夜 の 帐 に 包 まれ て ) January 19 , 2013 September 1 , 2015 After Yukino is expelled from Saber Tooth , she offers her two golden Gatekeys to Lucy to complete the latter 's collection of the Zodiac Gatekeys , stating that when the twelve Zodiac Gatekeys are all gathered , a gateway that changes the world will open ; however , Lucy turns down the offer . Meanwhile , Milliana tells Erza that Kagura 's sword , Archenemy , is intended to kill Jellal , and that she and Kagura share the same resentment towards Jellal . Natsu and Happy run after Yukino in an attempt to apologize ; however , Natsu realizes that Yukino had been expelled , and infiltrates the Crocus Garden to fight Jiemma . 166 `` Pandemonium '' `` Pandemoniumu '' ( 伏魔殿 ( パンデモニウム ) ) January 26 , 2013 September 1 , 2015 After Natsu punches out Dobengal , the guild member summoned to fight him , he and Jiemma engage in a duel . However , the guild master 's daughter Minerva appears , ends their duel and , taking Happy hostage , convinces Natsu to settle their dispute in the Grand Magic Games . As Natsu leaves , he states that Saber Tooth will never surpass Fairy Tail because they care for their friends , words that are echoed by Rogue , while Sting is left in awe at Natsu 's strength . The following day as the Grand Magic Games continue , an event called `` Pandemonium '' begins . It involves the participants entering a magical citadel filled with 100 monsters of varying strengths and individually doing battle with any number of monsters they choose . Representing Fairy Tail Team A , Erza requests the 100 monsters as her opponents . 167 `` 100 Against 1 '' `` Hyaku tai Ichi '' ( 100 対 1 ) February 2 , 2013 September 1 , 2015 Erza kills the 100 monsters using various types of armor to match their strength and weaknesses . As a result , the crowd praises Fairy Tail , and the guild is awarded with 10 points . Moreover , the game officials decided to determine the ranks of the remaining participants by means of a device called `` Magic Power Finder '' ( sometimes abbreviated as MPF ) . Using this tool , participants must attack it with their magic power , which will , in turn , calculate and display their equivalent scores depending on the extent of power used . In the end , Cana of Fairy Tail Team B dominates the event by using the Fairy Glitter spell lent to her by Mavis , much to Makarov 's surprise . With Cana 's victory , Fairy Tail Team B receives 8 points , and the audience began to burst once more into a resounding swarm of cheers and claps due to the finish of the two Fairy Tail teams . Meanwhile , the Raven Tail team somehow managed to get an insider information regarding the battle matchups for the day , so the masked leader Alexei tells his subordinates Flare , Obra , Nalpudding , and Kurohebi that it 's about time to proceed with their true mission . 168 `` Laxus vs. Alexei '' `` Rakusasu vs. Arekusei '' ( ラクサス vs. アレクセイ ) February 9 , 2013 September 1 , 2015 In the battle portion of the Grand Magic Games ' third day , Milliana of Mermaid Heel beats Semmes of Quatro Puppy while Saber Tooth 's Rufus swiftly defeats Blue Pegasus ' Eve . For the third match , Laxus of Fairy Tail Team B is chosen to battle against Alexei of Team Raven Tail . Since the opponent is from a suspicious guild , the Fairy Tail members take extra precautions by watching the actions of Ivan Dreyar as well as the other four Raven Tail members . However , as the match commences , the spectators are shocked to see Laxus being beaten by Alexei , although no cautious movements can be seen from the Raven Tail team . It turns out that the two dueling people are just illusions created by Alexei in order for him to have a talk with Laxus . Also , the other Raven Tail members being monitored by Fairy Tail are just lifelike illusions , for in reality they are with Alexei in the battlefield . Alexei reveals himself to be Ivan Dreyar . Unable to interrogate Laxus of the Lumen Histoire 's location , Ivan and the Raven Tail members are forced to fight him . Before their battle begins , Ivan declares that Raven Tail is an anti-Fairy Tail guild . 169 `` Wendy vs. Shelia '' `` Wendi vs. Sheria '' ( ウェンディ vs. シェリア ) February 16 , 2013 September 1 , 2015 After Laxus defeats Ivan and the four Raven Tail elite members , Ivan 's illusions disappear to the shock of everyone in the stadium . The game officials declare Laxus the victor , and Ivan tells him that Lumen Histoire is Fairy Tail 's ' darkness ' . Due to the infraction of the rules , the Raven Tail team is disqualified for the remainder of the Games , and its members are captured by the soldiers . Later , Wendy of Fairy Tail Team A and Sherria Blendy of Team Lamia Scale face off to compete for the final match of the day . As the two exchange attacks , gusts of wind surround the arena since Sherria is revealed to be a Sky God Slayer . Wendy employs an advanced Sky Dragon Slayer attack much to Poluchka 's astonishment , causing Sherria to fall on the ground . Wendy is about to be announced as the winner of the bout when Sherria suddenly stands up with her wounds fully healed . Meanwhile , the Crime Sorcière guild senses the nefarious aura , and Jellal proceeds to the Domus Flau audience , oblivious that he bumped into Doranbolt , who , in turn , wonders what is Mystogan doing in Earthland . 170 `` Small Fists '' `` Chiisana Kobushi '' ( 小さな 拳 ) February 23 , 2013 September 1 , 2015 With Cheria being fully healed and Wendy being almost depleted of magic power , their battle continues , with the former casting a high - level spell towards the latter . However , it seems that Wendy enhances Cheria 's physical strength , causing the attack to gain much force and thus missing Wendy . The two competitors then engage in a hand - to - hand combat , but due to the time limit and neither side is giving up , their battle resulted in a draw , giving both teams with 5 points . Meanwhile , Jellal manages to track the sinister aura among the crowd in the form of a mysterious , hooded person . Jellal is about to confront the aura source , but he is abruptly cornered by Doranbolt , Lahar , and other Rune Knights . Doranbalt then interrogates Jellal , and as his cover was blown , the Rune Knights , as well as Kagura and Milliana glancing nearby , are startled to see the true identity of Mystogan . Yajima convinces the Rune Knights that Mystogan is Jellal 's Edolas counterpart , prompting Lahar to release the fugitive . Meanwhile , Laxus and Makarov have a small conversation regarding Lumen Histoire when Mavis appears and explains that Lumen Histoire is the ' light ' of Fairy Tail . 171 `` Naval Battle '' `` Nabaru Batoru '' ( 海战 ( ナバルバトル ) ) March 2 , 2013 September 1 , 2015 As the jubilant Fairy Tail members celebrate their victories , the other people , such as Sting , Kagura , and Milliana , are contemplating on their own concerns , while Yukino is enlisted as a sergeant under Arcadios ' command . Meanwhile , the fourth day of the competitions comes to a start with an event , `` Naval Battle '' , which involves the participants swimming in a giant water sphere and knocking the other competitors out . After the other participants fall out of the aquatic sphere , a five - minute battle between Lucy and Minerva ensues . However , Minerva quickly attacks Lucy , and she manages to steal the keys of the Celestial Spirit mage . Regardless of this , Lucy refuses to surrender . After hearing Lucy 's intentions , Minerva stops from attacking up to the extent that the five - minute mark was reached . However , Minerva resumes attacking Lucy until the game officials decide to halt the match . The Fairy Tail members snap , and Minerva dangles Lucy out of the sphere while the others rush to her aid . 172 `` A Parfum for You '' `` Kimi ni Sasageru Parufamu '' ( 君 に 捧げる 香り ( パルファム ) ) March 9 , 2013 September 1 , 2015 While Lucy is hospitalized and taken to the infirmary , the Fairy Tail members warn Saber Tooth that they have angered the worst possible guild to do so . Later , Makarov announces that the two Fairy Tail teams must merge as a result of the disqualification of Raven Tail and to compensate for the odd number of teams for the battle round . He also reveals that the new team will have the former Team A 's 35 points to start with . The new Fairy Tail team , consisting of Natsu , Gray , Erza , Gajeel , and Laxus , are well - received as they enter the field . The second part of the tournament begins with a tag battle between Blue Pegasus ' Ichiya and `` Rabbit '' and Quatro Puppy 's Rocker and Bacchus . To everyone 's shock , it is revealed that `` Rabbit '' is actually Nichiya in disguise , and after the Exceed is punched by Bacchus , Ichiya bulks up to defeat the Quatro Puppy opponents , awarding Blue Pegasus with 10 points . 173 `` Battle of Dragon Slayers ! '' `` Batoru obu Doragon Sureiyā '' ( バトル ・ オブ ・ ドラゴン スレイヤー ) March 16 , 2013 September 1 , 2015 Mermaid Heel 's Kagura and Milliana and Lamia Scale 's Lyon and Yuka face off for the second tag battle of the games . The two teams struggle to earn a victory , only to have their battle ending in a draw due to the time limit . For the third and final match of the day , the spectators look on in silence as Natsu and Gajeel of Team Fairy Tail and Sting and Rogue of Team Saber Tooth enter the field . As the battle commences , Sting and Rogue were about to take the initiative when they were suddenly attacked by Natsu and Gajeel . The four mages then exchange attacks ; however , the audience watches in shock as the Fairy Tail Dragon Slayers overpower the Saber Tooth Dragon Slayers . Unable to keep themselves on par with their adversaries , Sting activates ' White Drive ' while Rogue powers up using ' Shadow Drive ' . 174 `` Four Dragons '' `` Yonin no Doragon '' ( 四 人 の 竜 ( ドラゴン ) ) March 23 , 2013 September 1 , 2015 The tag battle continues , with Sting and Rogue beginning to attack Natsu and Gajeel with their powers being increased . After receiving heavy blows from the Saber Tooth Dragon Slayers , Natsu and Gajeel manages to hold them off , thus forcing Sting and Rogue to activate `` Dragon Force '' on their own will , to the shock of the Fairy Tail audience . Sting decides to finish the Fairy Tail Dragon Slayers alone , for he held a promise to his Exceed friend Lector many years ago . As the three mages fight , the tide turns in favor of Sting as he unleashes a powerful attack on Natsu and Gajeel , resulting in the collapse of the arena floor , and after a hand - to - hand combat underground , Sting was seemingly the victor of the match . However , the Fairy Tail duo stands up , comments on Sting 's strength , and argues about his position while attacking , thereby causing Natsu to drag Gajeel on a mining cart and send it away . While Gajeel is left incapacitated due to motion sickness , Natsu challenges the Twin Dragons of Saber Tooth alone , in retribution to what Sting did earlier on them . 175 `` Natsu vs. the Twin Dragons '' `` Natsu vs. Sōryū '' ( ナツ vs. 双竜 ) March 30 , 2013 September 1 , 2015 The tag battle of the Dragon Slayers comes to a close , with Natsu overpowering the combined attacks of Sting and Rogue . The Saber Tooth members watch in dismay , while Jiemma seethes in anger over the defeat of the Twin Dragons , frightening the nearby spectators away . The rest of Fairy Tail celebrates in joy as its team receives 10 points , finally surpassing Team Saber Tooth by a single point . Meanwhile , Gajeel , back to his senses after being dragged by Natsu into a cart earlier , stumbles upon a cave full of dragon skeletons . During that night , Jellal manages to corner the source of the dark aura , standing in disbelief at the identity of the mysterious , hooded person .
  Fact 0.0: Episode 151, 'Sabertooth', has the Japanese title 'Seibatusu' and originally aired on October 6, 2012.
  Fact 0.1: Episode 151 aired in English on November 17, 2015.
  Fact 0.2: The Legion Platoon is tasked by Zentopia's archbishop to retrieve the missing pieces of the Infinity Clock and seal them again.
  Fact 0.3: Before embarking on their mission, the Legion Platoon visits Fairy Tail.
  Fact 0.4: The Legion Platoon makes amends with Fairy Tail for their previous hostility against them.
  Fact 0.5: It is later discovered that Fairy Tail has lost its title as the strongest guild in Fiore to Saber Tooth over the past seven years.
  Fact 0.6: Sting and Rogue are two of the strongest wizards in Fairy Tail.
  Fact 0.7: Sting and Rogue are a pair of Dragon Slayers.
  Fact 0.8: Sting is obsessed with fighting Natsu.
  Fact 0.9: Rogue is obsessed with fighting Gajeel.
  Fact 0.10: Makarov brings Gildarts to their abandoned guildhall.
  Fact 0.11: Makarov shows Gildarts Lumen Histoire, a light that Makarov says is Fairy Tail's greatest secret.
  Fact 0.12: Makarov tells Gildarts that he has been chosen to become the next master of Fairy Tail.
  Fact 0.13: Episode 152, 'And So We Aim for the Top', has the Japanese title 'Soshite Ore-tachi wa Chojo o Mezasu' and originally aired on October 13, 2012.
  Fact 0.14: Episode 152 aired in English on November 24, 2015.
  Fact 0.15: Natsu and his friends discover that a power gap has developed between them and the rest of the guild members who were not on Tenrou Island.
  Fact 0.16: The group meets with Poluchka for medicine to strengthen their magic.
  Fact 0.17: The group is unsuccessful in strengthening their magic with the medicine.
  Fact 0.18: Wendy recognizes Poluchka's voice and scent as those of Grandine, the dragon who raised her.
  Fact 0.19: Poluchka reveals herself as the Edolas counterpart of Grandine.
  Fact 0.20: Poluchka gives Wendy written instructions for two advanced spells that Grandine failed to teach her.
  Fact 0.21: Makarov announces Gildarts' succession as master to the rest of the guild.
  Fact 0.22: Makarov finds that Gildarts has gone on a journey, leaving a letter of resignation behind.
  Fact 0.23: In the letter, Gildarts reinstates Laxus into the guild.
  Fact 0.24: In the letter, Gildarts returns mastership to Makarov.
  Fact 0.25: In the letter, Gildarts tells Makarov to make Fairy Tail the strongest guild in Fiore again.
  Fact 0.26: Romeo suggests they participate in the Grand Magic Games.
  Fact 0.27: The Grand Magic Games are an annual festival designed to determine the guild rankings in Fiore.
  Fact 0.28: Episode 153, 'Song of the Stars', has the Japanese title 'Hoshiboshi no Uta' and originally aired on October 20, 2012.
  Fact 0.29: Episode 153 aired in English on December 1, 2015.
  Fact 0.30: The Grand Magic Games are occurring in three months' time.
  Fact 0.31: Natsu and his friends train at a beach resort.
  Fact 0.32: On their second day of training, Virgo appears and tells the wizards that the Celestial Spirit World is in danger of destruction.
  Fact 0.33: When the wizards arrive, they find a surprise party held by the Celestial Spirit King.
  Fact 0.34: The party celebrates the wizards' return from Sirius Island after seven years.
  Fact 0.35: The wizards spend the day in the spirit world partying with the spirits.
  Fact 0.36: Virgo informs the wizards that one day in the spirit world equals three months in the human world.
  Fact 0.37: The wizards find that the Grand Magic Games are upon them.
  Fact 0.38: Episode 154, 'For All the Time We Missed Each Other', has the Japanese title 'Surechigatta Jikan no Bun dake' and originally aired on October 27, 2012.
  Fact 0.39: Episode 154 aired in English on December 8, 2015.
  Fact 0.40: While mulling over their lost training time, the Fairy Tail wizards receive a message via carrier pigeon.
  Fact 0.41: The message requests a private meeting in the forest nearby.
  Fact 0.42: At the meeting, the wizards meet a runaway Jellal, Ultear, and Merudy.
  Fact 0.43: Jellal, Ultear, and Merudy have formed their own guild, Crime Sorciere.
  Fact 0.44: Crime Sorciere aims to defeat Zeref and any dark guilds that try to use his power.
  Fact 0.45: Jellal, Ultear, and Merudy ask Fairy Tail for help in determining the source of a magical anomaly similar to Zeref's magic.
  Fact 0.46: The magical anomaly has been sensed around the Grand Magic Games arena, Domus Flau.
  Fact 0.47: In exchange, Ultear uses her magic to invoke Fairy Tail's Second Origin in order to double their magic power.
  Fact 0.48: Erza talks with Jellal over his regained memories.
  Fact 0.49: Erza and Jellal nearly kiss.
  Fact 0.50: Jellal decides that he and Erza can not be together.
  Fact 0.51: Jellal lies to Erza that he has a fiancee.
  Fact 0.52: Erza quickly picks up on the lie and respects his decision.
  Fact 0.53: Erza sees Crime Sorciere off as they leave.
  Fact 0.54: Episode 155, 'Crocus, the Flower-Blooming Capital', has the Japanese title 'Hanasaku Miyako: Kurokkasu' and originally aired on November 3, 2012.
  Fact 0.55: Episode 155 aired in English on December 15, 2015.
  Fact 0.56: Natsu, Lucy, Gray, Erza, and Wendy are chosen to represent their guild in the Grand Magic Games.
  Fact 0.57: The Grand Magic Games are to be held in Fiore's capital, Crocus.
  Fact 0.58: As they tour the city, Natsu and Happy encounter the twin dragon slayers, Sting and Rogue.
  Fact 0.59: Sting and Rogue mock Natsu for not being able to kill Acnologia.
  Fact 0.60: Sting and Rogue boast of becoming true dragon slayers by killing the dragons who raised them.
  Fact 0.61: At midnight, Fairy Tail and the other participating guilds are put through a preliminary event.
  Fact 0.62: The preliminary event is a race through a massive floating labyrinth.
  Fact 0.63: The preliminary event is designed to narrow the number of participating guilds down to eight teams.
  Fact 0.64: Episode 156, 'Sky Labyrinth', has the Japanese title 'Sukai Rabirinsu' and originally aired on November 10, 2012.
  Fact 0.65: Episode 156 aired in English on December 22, 2015.
  Fact 0.66: Natsu's team notices that Wendy has gone missing.
  Fact 0.67: Elfman replaces Wendy to keep Natsu's team from being eliminated.
  Fact 0.68: As they make their way through the enchanted labyrinth, the five encounter various other guilds.
  Fact 0.69: The five steal the maps made by the other guilds to better navigate the maze.
  Fact 0.70: The five reach the end of the maze and pass the elimination round.
  Fact 0.71: The five are confident they have arrived first.
  Fact 0.72: The five discover they have barely taken last place.
  Fact 0.73: Happy, Lisanna, and the rest of the guild search for Wendy and Carla.
  Fact 0.74: Happy, Lisanna, and the rest of the guild find Wendy and Carla in the garden of the king's palace, Mercurius.
  Fact 0.75: Wendy and Carla have their magic power depleted.
  Fact 0.76: Episode 157, 'New Guild', has the Japanese title 'Shinki Girudo' and originally aired on November 17, 2012.
  Fact 0.77: Episode 157 aired in English on December 29, 2015.
  Fact 0.78: Wendy's body is weakened as a result of losing her magic power.
  Fact 0.79: Poluchka helps Wendy recover.
  Fact 0.80: Wendy formally requests Elfman to take her place in the Grand Magic Games.
  Fact 0.81: Fairy Tail steps into the arena with a negative audience reception.
  Fact 0.82: Fairy Tail is bolstered by their guild mates and the spirit of Mavis, whom only Fairy Tail guild members can see.
  Fact 0.83: The next four participating teams are Quatro Cerberus, Mermaid Heel, Blue Pegasus, and Lamia Scale.
  Fact 0.84: Raven Tail is a former dark guild led by Makarov's son Ivan.
  Fact 0.85: Raven Tail members confess to attacking Wendy and Carla.
  Fact 0.86: The next team is a second team from Fairy Tail composed of Gajeel, Juvia, Mirajane, Laxus, and Jellal.
  Fact 0.87: Jellal is disguised as Mystogan to help investigate the arena.
  Fact 0.88: The final guild to be announced is Saber Tooth.
  Fact 0.89: The audience cheers joyfully for Saber Tooth's announcement.
  Fact 0.90: All eight teams are announced.
  Fact 0.91: Each team sends out one member to participate in the first contest of the games.
  Fact 0.92: Gray and Juvia represent their teams in the first contest.
  Fact 0.93: Episode 158, 'Night of Shooting Stars', has the Japanese title 'Hoshi Furu Yoru ni' and originally aired on November 24, 2012.
  Fact 0.94: Episode 158 aired in English on January 5, 2016.
  Fact 0.95: The games' first contest, 'Hidden', requires the eight participants to score points by finding and attacking each other.
  Fact 0.96: The contest takes place in a virtual city filled with copies of the participants.
  Fact 0.97: Raven Tail member Nalpudding repeatedly targets Gray.
  Fact 0.98: Gray scores last place in the contest.
  Fact 0.99: Saber Tooth's Rufus scores first by using his Memory-Make magic to attack all the other participants at once.
  Fact 0.100: Gray is humiliated and swears revenge against Raven Tail and Saber Tooth.
  Fact 0.101: Lucy is chosen to fight Raven Tail member Flare in the tournament's first battle round.
  Fact 0.102: Episode 159, 'Lucy vs. Flare', has the Japanese title 'Rushii vs. Furea' and originally aired on December 1, 2012.
  Fact 0.103: Episode 159 aired in English on January 12, 2016.
  Fact 0.104: Lucy displays tremendous skill in her battle against Flare by summoning two spirits simultaneously.
  Fact 0.105: Flare cheats by using her magic hair to hold Alzack and Bisca's daughter, Asuka, as a hostage.
  Fact 0.106: Flare blackmails Lucy into not fighting back to keep Flare from harming Asuka.
  Fact 0.107: Lucy tearfully speaks of Asuka's danger.
  Fact 0.108: Natsu hears Lucy and rushes to Asuka.
  Fact 0.109: Natsu incinerates Flare's hair.
  Fact 0.110: Lucy counterattacks with her Urano Metoria spell in combination with Gemini.
  Fact 0.111: Flare's teammate Obra nullifies Lucy's magic before the spell fully activates.
  Fact 0.112: Lucy collapses and loses the match.
  Fact 0.113: Episode 160, 'Portent', has the Japanese title 'Kyozui' and originally aired on December 8, 2012.
  Fact 0.114: Episode 160 aired in English on January 19, 2016.
  Fact 0.115: Blue Pegasus' Ren defeats Mermaid Heel's Arana during the games' second match.
  Fact 0.116: Saber Tooth's Orga effortlessly defeats Quatro Cerberus' War Cry in the third match.
  Fact 0.117: For the day's fourth and final match, Jellal is pitted against Lamia Scale's Jura.
  Fact 0.118: Jellal intends to avenge the humiliated Fairy Tail members.
  Fact 0.119: Jellal mimics Mystogan's magic to keep up his disguise.
  Fact 0.120: Jellal finds Mystogan's magic ineffective against Jura.
  Fact 0.121: Jellal is forced to use his own magic instead.
  Fact 0.122: Before Jellal can cast a spell that will expose his true identity, Ultear uses Merudy's sensory to force Jellal into submission.
  Fact 0.123: The hospitalized Carla awakens after having a troubling premonition of the future.
  Fact 0.124: Episode 161, 'Chariots', has the Japanese title 'Chariotto' and originally aired on December 15, 2012.
  Fact 0.125: Episode 161 aired in English on January 26, 2016.
  Fact 0.126: Fairy Tail is undaunted by their poor performance on the first day.
  Fact 0.127: As the guild celebrates, they are visited and taunted by Bacchus.
  Fact 0.128: Bacchus is Erza's rival from Quatro Cerberus.
  Fact 0.129: Bacchus has come to replace War Cry.
  Fact 0.130: Carla relays her premonition to Poluchka.
  Fact 0.131: In Carla's premonition, Mercurius is collapsing around what appeared to be a tearfully singing Lucy.
  Fact 0.132: The second day of the tournament begins with a footrace atop a train of moving chariots.
  Fact 0.133: Natsu, Gajeel, and Sting suffer from motion sickness as a side effect of their Dragon Slayer magic.
  Fact 0.134: Natsu responds to Sting that his determination is a result of the strength of Fairy Tail's bonds.
  Fact 0.135: Sting eventually throws the race.
  Fact 0.136: Natsu finishes as sixth place.
  Fact 0.137: Gajeel finishes as seventh place.
  Fact 0.138: Natsu and Gajeel earn Fairy Tail the audience's respect.
  Fact 0.139: Episode 162, 'Elfman vs. Bacchus', has the Japanese title 'Erufuman vs. Bakkasu' and originally aired on December 22, 2012.
  Fact 0.140: Episode 162 aired in English on September 1, 2015.
  Fact 0.141: Due to a miscommunication between the king and his knight Arcadios, Elfman is chosen as Bacchus' opponent instead of Erza.
  Fact 0.142: Before the match, Bacchus and Elfman make a wager.
  Fact 0.143: If Bacchus wins, he will court Elfman's sisters.
  Fact 0.144: If Elfman wins, Bacchus' team name will be changed to 'Quatro Puppy' for the rest of the games.
  Fact 0.145: Bacchus ruthlessly attacks Elfman with his bare palms.
  Fact 0.146: Elfman damages Bacchus with his defensive Lizardman transformation.
  Fact 0.147: Elfman outlasts Bacchus as Bacchus exhausts himself.
  Fact 0.148: Elfman wins the match.
  Fact 0.149: Bacchus' team is renamed.
  Fact 0.150: Natsu finds Wendy, Carla, and Poluchka being kidnapped by a group of masked men.
  Fact 0.151: The masked men claim they were hired to kidnap Lucy by Raven Tail.
  Fact 0.152: The masked men are revealed to have actually been sent by Arcadios.
  Fact 0.153: Arcadios intends to sow discord between the guilds.
  Fact 0.154: Arcadios plots to use Lucy for his 'Eclipse Project'.
  Fact 0.155: Episode 163, 'Mirajane vs. Jenny', has the Japanese title 'Mirajen vs. Jeni' and originally aired on January 5, 2013.
  Fact 0.156: Episode 163 aired in English on September 1, 2015.
  Fact 0.157: The second day of the Grand Magic Games continues.
  Fact 0.158: Mirajane of Fairy Tail Team B fights against Jenny Realight of Team Blue Pegasus.
  Fact 0.159: Mirajane and Jenny are well-known for being magazine models.
  Fact 0.160: Their battle consists almost entirely of a fashion showdown.
  Fact 0.161: Jenny proposes a wager that the loser must pose nude for the Sorcerer Magazine.
  Fact 0.162: Mirajane accepts the wager.
  Fact 0.163: Jenny transforms into her battle armor.
  Fact 0.164: Mirajane shifts into her Satan Soul 'Sitri' form.
  Fact 0.165: Mirajane easily defeats Jenny.
  Fact 0.166: The final match of the day is Kagura Mikazuchi of Mermaid Heel against Yukino Aguria of Saber Tooth.
  Fact 0.167: Yukino proposes to Kagura that they wager their lives on their battle.
  Fact 0.168: Arcadios is confronted by Darton, the kingdom's defense minister, about his 'Eclipse Project'.
  Fact 0.169: Episode 164, 'Kagura vs. Yukino', has the Japanese title 'Kagura vs. Yukino' and originally aired on January 12, 2013.
  Fact 0.170: Episode 164 aired in English on September 1, 2015.
  Fact 0.171: Yukino is revealed to be a celestial spirit wizard when she summons Libra and Pisces to attack Kagura.
  Fact 0.172: Kagura defeats Libra and Pisces using her gravity-altering magic.
  Fact 0.173: Yukino uses a Gatekey that summons Ophiuchus, which is regarded as the strongest among the Celestial Spirits.
  Fact 0.174: Kagura slays Ophiuchus with her sword still sheathed.
  Fact 0.175: Kagura easily defeats Yukino.
  Fact 0.176: Jellal discusses the fact that he has not sensed the evil power similar to Zeref's.
  Fact 0.177: Jellal's discussion takes place after the second day of the Grand Magic Games has ended.
  Fact 0.178: Erza reunites with Milliana.
  Fact 0.179: Milliana is revealed to be a member of Mermaid Heel.
  Fact 0.180: Saber Tooth's master Jiemma is dissatisfied with his guild's performance.
  Fact 0.181: Sting is given another chance.
  Fact 0.182: Jiemma humiliates Yukino for her failure.
  Fact 0.183: Episode 165, 'Hatred at Nightfall', has the Japanese title 'Urami wa Yoru no Tobari ni Tsutsumarete' and originally aired on January 19, 2013.
  Fact 0.184: Episode 165 aired in English on September 1, 2015.
  Fact 0.185: After Yukino is expelled from Saber Tooth, she offers her two golden Gatekeys to Lucy.
  Fact 0.186: Yukino offers the Gatekeys to complete Lucy's collection of the Zodiac Gatekeys.
  Fact 0.187: Yukino states that when the twelve Zodiac Gatekeys are all gathered, a gateway that changes the world will open.
  Fact 0.188: Lucy turns down the offer.
  Fact 0.189: Milliana tells Erza that Kagura's sword, Archenemy, is intended to kill Jellal.
  Fact 0.190: Milliana tells Erza that she and Kagura share the same resentment towards Jellal.
  Fact 0.191: Natsu and Happy run after Yukino in an attempt to apologize.
  Fact 0.192: Natsu realizes that Yukino had been expelled.
  Fact 0.193: Natsu infiltrates the Crocus Garden to fight Jiemma.
  Fact 0.194: Episode 166, 'Pandemonium', has the Japanese title 'Pandemoniumu' and originally aired on January 26, 2013.
  Fact 0.195: Episode 166 aired in English on September 1, 2015.
  Fact 0.196: Natsu punches out Dobengal, the guild member summoned to fight him.
  Fact 0.197: Natsu and Jiemma engage in a duel.
  Fact 0.198: The guild master's daughter, Minerva, appears and ends their duel.
  Fact 0.199: Minerva takes Happy hostage.
  Fact 0.200: Minerva convinces Natsu to settle their dispute in the Grand Magic Games.
  Fact 0.201: As Natsu leaves, he states that Saber Tooth will never surpass Fairy Tail because they care for their friends.
  Fact 0.202: Rogue echoes Natsu's words.
  Fact 0.203: Sting is left in awe at Natsu's strength.
  Fact 0.204: The following day, an event called 'Pandemonium' begins.
  Fact 0.205: In Pandemonium, participants enter a magical citadel filled with 100 monsters of varying strengths.
  Fact 0.206: Participants individually do battle with any number of monsters they choose.
  Fact 0.207: Erza, representing Fairy Tail Team A, requests the 100 monsters as her opponents.
  Fact 0.208: Episode 167, '100 Against 1', has the Japanese title '100 tai 1' and originally aired on February 2, 2013.
  Fact 0.209: Episode 167 aired in English on September 1, 2015.
  Fact 0.210: Erza kills the 100 monsters using various types of armor to match their strengths and weaknesses.
  Fact 0.211: The crowd praises Fairy Tail.
  Fact 0.212: Fairy Tail is awarded 10 points.
  Fact 0.213: The game officials decide to determine the ranks of the remaining participants by means of a device called the 'Magic Power Finder', sometimes abbreviated as MPF.
  Fact 0.214: Participants must attack the Magic Power Finder with their magic power.
  Fact 0.215: The Magic Power Finder calculates and displays participants' equivalent scores based on the extent of power used.
  Fact 0.216: Cana of Fairy Tail Team B dominates the event by using the Fairy Glitter spell lent to her by Mavis.
  Fact 0.217: Makarov is surprised by Cana's use of the Fairy Glitter spell.
  Fact 0.218: Fairy Tail Team B receives 8 points.
  Fact 0.219: The audience bursts into a resounding swarm of cheers and claps due to the finish of the two Fairy Tail teams.
  Fact 0.220: The masked leader Alexei tells his subordinates Flare, Obra, Nalpudding, and Kurohebi that it is about time to proceed with their true mission.
  Fact 0.221: Raven Tail somehow got insider information regarding the battle matchups for the day.
  Fact 0.222: Episode 168, 'Laxus vs. Alexei', has the Japanese title 'Rakusasu vs. Arekusei' and originally aired on February 9, 2013.
  Fact 0.223: Episode 168 aired in English on September 1, 2015.
  Fact 0.224: In the battle portion of the Grand Magic Games' third day, Milliana of Mermaid Heel beats Semmes of Quatro Puppy.
  Fact 0.225: Saber Tooth's Rufus swiftly defeats Blue Pegasus' Eve.
  Fact 0.226: Laxus of Fairy Tail Team B is chosen to battle against Alexei of Team Raven Tail.
  Fact 0.227: The Fairy Tail members take extra precautions because Raven Tail is a suspicious guild.
  Fact 0.228: Fairy Tail watches the actions of Ivan Dreyar and the other four Raven Tail members.
  Fact 0.229: The spectators are shocked to see Laxus being beaten by Alexei.
  Fact 0.230: No cautious movements can be seen from the Raven Tail team.
  Fact 0.231: The two dueling people are illusions created by Alexei.
  Fact 0.232: Alexei created the illusions so that he could have a talk with Laxus.
  Fact 0.233: The other Raven Tail members being monitored by Fairy Tail are lifelike illusions.
  Fact 0.234: In reality, the other Raven Tail members are with Alexei on the battlefield.
  Fact 0.235: Alexei reveals himself to be Ivan Dreyar.
  Fact 0.236: Ivan and the Raven Tail members are unable to interrogate Laxus about the Lumen Histoire's location.
  Fact 0.237: Ivan and the Raven Tail members are forced to fight Laxus.
  Fact 0.238: Before their battle begins, Ivan declares that Raven Tail is an anti-Fairy Tail guild.
  Fact 0.239: Episode 169, 'Wendy vs. Shelia', has the Japanese title 'Wendi vs. Sheria' and originally aired on February 16, 2013.
  Fact 0.240: Episode 169 aired in English on September 1, 2015.
  Fact 0.241: Laxus defeats Ivan and the four Raven Tail elite members.
  Fact 0.242: Ivan's illusions disappear to the shock of everyone in the stadium.
  Fact 0.243: The game officials declare Laxus the victor.
  Fact 0.244: Ivan tells Laxus that Lumen Histoire is Fairy Tail's 'darkness'.
  Fact 0.245: Raven Tail is disqualified for the remainder of the Games due to the infraction of the rules.
  Fact 0.246: Raven Tail's members are captured by the soldiers.
  Fact 0.247: Wendy of Fairy Tail Team A and Sherria Blendy of Team Lamia Scale face off for the final match of the day.
  Fact 0.248: Sherria is revealed to be a Sky God Slayer.
  Fact 0.249: Gusts of wind surround the arena as Sherria and Wendy exchange attacks.
  Fact 0.250: Wendy employs an advanced Sky Dragon Slayer attack, much to Poluchka's astonishment.
  Fact 0.251: Sherria falls on the ground.
  Fact 0.252: Wendy is about to be announced as the winner of the bout.
  Fact 0.253: Sherria suddenly stands up with her wounds fully healed.
  Fact 0.254: The Crime Sorciere guild senses the nefarious aura.
  Fact 0.255: Jellal proceeds to the Domus Flau audience.
  Fact 0.256: Jellal is oblivious that he bumped into Doranbolt.
  Fact 0.257: Doranbolt wonders what Mystogan is doing in Earthland.
  Fact 0.258: Episode 170, 'Small Fists', has the Japanese title 'Chiisana Kobushi' and originally aired on February 23, 2013.
  Fact 0.259: Episode 170 aired in English on September 1, 2015.
  Fact 0.260: Cheria is fully healed.
  Fact 0.261: Wendy is almost depleted of magic power.
  Fact 0.262: Cheria casts a high-level spell towards Wendy.
  Fact 0.263: Wendy enhances Cheria's physical strength, causing the attack to miss Wendy.
  Fact 0.264: The two competitors engage in hand-to-hand combat.
  Fact 0.265: Due to the time limit and neither side giving up, their battle results in a draw.
  Fact 0.266: The draw gives both teams 5 points.
  Fact 0.267: Jellal tracks the sinister aura among the crowd in the form of a mysterious, hooded person.
  Fact 0.268: Jellal is about to confront the aura source.
  Fact 0.269: Doranbolt, Lahar, and other Rune Knights corner Jellal.
  Fact 0.270: Doranbolt interrogates Jellal.
  Fact 0.271: Jellal's cover is blown.
  Fact 0.272: The Rune Knights, Kagura and Milliana see the true identity of Mystogan.
  Fact 0.273: Yajima convinces the Rune Knights that Mystogan is Jellal's Edolas counterpart.
  Fact 0.274: Lahar releases the fugitive.
  Fact 0.275: Laxus and Makarov have a small conversation regarding Lumen Histoire.
  Fact 0.276: Mavis appears and explains that Lumen Histoire is the 'light' of Fairy Tail.
  Fact 0.277: Episode 171, 'Naval Battle', has the Japanese title 'Nabaru Batoru' and originally aired on March 2, 2013.
  Fact 0.278: Episode 171 aired in English on September 1, 2015.
  Fact 0.279: The jubilant Fairy Tail members celebrate their victories.
  Fact 0.280: Sting, Kagura, and Milliana are contemplating their own concerns.
  Fact 0.281: Yukino is enlisted as a sergeant under Arcadios' command.
  Fact 0.282: The fourth day of the competitions begins with an event called 'Naval Battle'.
  Fact 0.283: Naval Battle involves participants swimming in a giant water sphere and knocking the other competitors out.
  Fact 0.284: After the other participants fall out of the sphere, a five-minute battle between Lucy and Minerva ensues.
  Fact 0.285: Minerva quickly attacks Lucy and steals the keys of the Celestial Spirit mage.
  Fact 0.286: Lucy refuses to surrender.
  Fact 0.287: After hearing Lucy's intentions, Minerva stops attacking until the five-minute mark is reached.
  Fact 0.288: Minerva resumes attacking Lucy until the game officials decide to halt the match.
  Fact 0.289: The Fairy Tail members snap.
  Fact 0.290: Minerva dangles Lucy out of the sphere.
  Fact 0.291: The other Fairy Tail members rush to Lucy's aid.
  Fact 0.292: Episode 172, 'A Parfum for You', has the Japanese title 'Kimi ni Sasageru Parufamu' and originally aired on March 9, 2013.
  Fact 0.293: Episode 172 aired in English on September 1, 2015.
  Fact 0.294: Lucy is hospitalized and taken to the infirmary.
  Fact 0.295: The Fairy Tail members warn Saber Tooth that they have angered the worst possible guild to do so.
  Fact 0.296: Makarov announces that the two Fairy Tail teams must merge.
  Fact 0.297: The merger is a result of the disqualification of Raven Tail.
  Fact 0.298: The merger compensates for the odd number of teams for the battle round.
  Fact 0.299: The new team will have the former Team A's 35 points to start with.
  Fact 0.300: The new Fairy Tail team consists of Natsu, Gray, Erza, Gajeel, and Laxus.
  Fact 0.301: The new team is well-received as they enter the field.
  Fact 0.302: The second part of the tournament begins with a tag battle between Blue Pegasus' Ichiya and 'Rabbit' and Quatro Puppy's Rocker and Bacchus.
  Fact 0.303: 'Rabbit' is revealed to be Ichiya in disguise.
  Fact 0.304: Bacchus punches the Exceed.
  Fact 0.305: Ichiya bulks up to defeat the Quatro Puppy opponents.
  Fact 0.306: Blue Pegasus is awarded 10 points.
  Fact 0.307: Episode 173, 'Battle of Dragon Slayers!', has the Japanese title 'Batoru obu Doragon Sureiya' and originally aired on March 16, 2013.
  Fact 0.308: Episode 173 aired in English on September 1, 2015.
  Fact 0.309: Mermaid Heel's Kagura and Milliana face off against Lamia Scale's Lyon and Yuka for the second tag battle of the games.
  Fact 0.310: The second tag battle ends in a draw due to the time limit.
  Fact 0.311: For the third and final match of the day, Natsu and Gajeel of Team Fairy Tail enter the field against Sting and Rogue of Team Saber Tooth.
  Fact 0.312: Sting and Rogue were about to take the initiative when Natsu and Gajeel attacked them.
  Fact 0.313: The audience watches in shock as the Fairy Tail Dragon Slayers overpower the Saber Tooth Dragon Slayers.
  Fact 0.314: Sting activates 'White Drive'.
  Fact 0.315: Rogue powers up using 'Shadow Drive'.
  Fact 0.316: Episode 174, 'Four Dragons', has the Japanese title 'Yonin no Doragon' and originally aired on March 23, 2013.
  Fact 0.317: Episode 174 aired in English on September 1, 2015.
  Fact 0.318: Sting and Rogue begin attacking Natsu and Gajeel with their powers increased.
  Fact 0.319: After receiving heavy blows from the Saber Tooth Dragon Slayers, Natsu and Gajeel manage to hold them off.
  Fact 0.320: Sting and Rogue are forced to activate 'Dragon Force' on their own will.
  Fact 0.321: The Fairy Tail audience is shocked by Sting and Rogue activating 'Dragon Force'.
  Fact 0.322: Sting decides to finish the Fairy Tail Dragon Slayers alone.
  Fact 0.323: Sting held a promise to his Exceed friend Lector many years ago.
  Fact 0.324: The tide turns in favor of Sting as he unleashes a powerful attack on Natsu and Gajeel.
  Fact 0.325: Sting's attack causes the collapse of the arena floor.
  Fact 0.326: After a hand-to-hand combat underground, Sting is seemingly the victor of the match.
  Fact 0.327: Natsu and Gajeel stand up, comment on Sting's strength, and argue about his position while attacking.
  Fact 0.328: Natsu drags Gajeel on a mining cart and sends it away.
  Fact 0.329: Gajeel is left incapacitated due to motion sickness.
  Fact 0.330: Natsu challenges the Twin Dragons of Saber Tooth alone, in retribution for what Sting did earlier.
  Fact 0.331: Episode 175, 'Natsu vs. the Twin Dragons', has the Japanese title 'Natsu vs. Soryu' and originally aired on March 30, 2013.
  Fact 0.332: Episode 175 aired in English on September 1, 2015.
  Fact 0.333: The tag battle of the Dragon Slayers comes to a close.
  Fact 0.334: Natsu overpowers the combined attacks of Sting and Rogue.
  Fact 0.335: The Saber Tooth members watch in dismay.
  Fact 0.336: Jiemma seethes in anger over the defeat of the Twin Dragons.
  Fact 0.337: Jiemma's anger frightens the nearby spectators away.
  Fact 0.338: The rest of Fairy Tail celebrates in joy as its team receives 10 points.
  Fact 0.339: Fairy Tail surpasses Team Saber Tooth by a single point.
  Fact 0.340: Gajeel, back to his senses after being dragged by Natsu into a cart earlier, stumbles upon a cave full of dragon skeletons.
  Fact 0.341: That night, Jellal manages to corner the source of the dark aura.
  Fact 0.342: Jellal stands in disbelief at the identity of the mysterious, hooded person.

Retrieved items:
<item 1>
List of Fairy Tail episodes (season 6)
The following day as the Grand Magic Games continue , an event called `` Pandemonium '' begins . It involves the participants entering a magical citadel filled
with 100 monsters of varying strengths and individually doing battle with any number of monsters they choose . Representing Fairy Tail Team A , Erza requests
the 100 monsters as her opponents . 167 `` 100 Against 1 '' `` Hyaku tai Ichi '' ( 100 対 1 ) February 2 , 2013 September 1 , 2015 Erza kills the 100 monsters
using various types of armor to match their strength and weaknesses . As a result , the crowd praises Fairy Tail , and the guild is awarded with 10 points .
Moreover , the game officials decided to determine the ranks of the remaining participants by means of a device called `` Magic Power Finder '' ( sometimes
abbreviated as MPF ) . Using this tool , participants must attack it with their magic power , which will , in turn , calculate and display their equivalent
scores depending on the extent of power used .
</item 1>

<item 2>
List of Fairy Tail episodes (season 6)
Poluchka reveals herself as the Edolas counterpart of Grandine , and gives Wendy written instructions for two advanced spells that Grandine failed to teach her
. Meanwhile , Makarov announces Gildarts ' succession as master to the rest of the guild , only to find he has gone on a journey , leaving a letter of
resignation behind . In the letter , Gildarts reinstates Laxus into the guild , returns mastership to Makarov , and tells him to make Fairy Tail the strongest
guild in Fiore again . Romeo suggests they participate in the Grand Magic Games , an annual festival designed to determine the guild rankings in Fiore . 153 ``
Song of the Stars '' `` Hoshiboshi no Uta '' ( 星々 の 歌 ) October 20 , 2012 December 1 , 2015 With the Grand Magic Games occurring in three months time , Natsu
and his friends train at a beach resort . On their second day of training , Virgo appears and tells the wizards that the Celestial Spirit World is in danger of
destruction .
</item 2>

<item 3>
List of Fairy Tail episodes (season 6)
It contains a single story arc , titled `` Dai Matō Enbu '' ( 大 魔 闘 演武 , lit `` Grand Magic Games '' ) , which adapts material from the beginning of the 31st
through the ending of the 35th volumes of Hiro Mashima 's Fairy Tail manga over 24 episodes . Focusing on Natsu and the others who were frozen in time for seven
years on Tenrou Island , the members return to Magnolia and discover that Saber Tooth is the strongest guild in the Fiore Kingdom . Having become the weakest
guild during the absence of its core members , Fairy Tail decides to enter the Grand Magic Games , an annual competition to decide Fiore 's strongest guild .
The season began airing on October 6 , 2012 on TV Tokyo in Japan and concluded its run on March 30 , 2013 . The first DVD compilation , containing the last two
episodes of the fifth season and the first two episodes of the sixth season , was released by Pony Canyon on March 6 , 2013 .
</item 3>

<item 4>
List of Fairy Tail episodes (season 6)
To everyone 's shock , the next team is revealed to be a second team from Fairy Tail composed of Gajeel , Juvia , Mirajane , Laxus , and Jellal ( disguised as
Mystogan to help investigate the arena ) . The final guild to be announced is Saber Tooth , much to the joy of the audience . With all eight teams announced ,
they each send out one member to participate in the first contest of the games , with Gray and Juvia representing their teams . 158 `` Night of Shooting Stars
'' `` Hoshi Furu Yoru ni '' ( 星 降 ル 夜 ニ ) November 24 , 2012 January 5 , 2016 The games ' first contest , `` Hidden '' , requires the eight participants to
score points by finding and attacking each other in a virtual city filled with copies of themselves . Raven Tail member Nalpudding repeatedly targets Gray , who
scores last place in the contest while Saber Tooth 's Rufus scores first by using his Memory - Make magic to attack all the other participants at once .
</item 4>

<item 5>
List of Fairy Tail episodes (season 6)
Meanwhile , Natsu finds Wendy , Carla , and Poluchka being kidnapped by a group of masked men . Upon being caught , the men claim they were hired to kidnap Lucy
by Raven Tail . They are soon revealed to have actually been sent by Arcadios , who intends to sow discord between the guilds as he plots to use Lucy for his ``
Eclipse Project '' . 163 `` Mirajane vs. Jenny '' `` Mirajēn vs. Jenī '' ( ミラ ジェーン vs. ジェニー ) January 5 , 2013 September 1 , 2015 The second day of the Grand
Magic Games continues , with Mirajane of Fairy Tail Team B fighting against Jenny Realight of Team Blue Pegasus . Since the two competitors are well - known for
being magazine models , their battle consists almost entirely of a fashion showdown , much to the pleasure of the audience . Eventually as the showdown becomes
redundant , Jenny proposes a wager : the loser must pose nude for the Sorcerer Magazine , which Mirajane accepts .
</item 5>

<item 6>
List of Fairy Tail episodes (season 6)
Meanwhile , Happy , Lisanna , and the rest of the guild search for Wendy and Carla . They eventually find the two in the garden of the king 's palace ,
Mercurius , with their magic power depleted . 157 `` New Guild '' `` Shinki Girudo '' ( 新 规 ギルド ) November 17 , 2012 December 29 , 2015 Wendy 's body is
weakened as a result of losing her magic power . As Poluchka helps her recover , she formally requests Elfman to take her place in the Grand Magic Games . Fairy
Tail steps into the arena with a negative audience reception , but they are bolstered by their guild mates and the spirit of Mavis , whom only the Fairy Tail
guild members can see . The next four participating teams are revealed to be Quatro Cerberus , Mermaid Heel , Blue Pegasus , and Lamia Scale . Following them is
Raven Tail , a former dark guild led by Makarov 's son Ivan , who confess to attacking Wendy and Carla .
</item 6>

<item 7>
List of Fairy Tail episodes (season 6)
The three ask Fairy Tail for their help in determining the source of a magical anomaly similar to Zeref 's magic that they have sensed around the Grand Magic
Games arena , Domus Flau . In exchange , Ultear uses her magic to invoke their Second Origin in order to double their magic power . As the others undergo the
painful process of awakening their Second Origin , Erza talks with Jellal over his regained memories . The two nearly kiss , but Jellal decides that he and Erza
can not be together and lies to her that he has a fiancée . Erza quickly picks up on this and respects his decision , seeing Crime Sorcière off as they leave .
155 `` Crocus , the Flower - Blooming Capital '' `` Hanasaku Miyako : Kurokkasu '' ( 花 笑 く 都 ・ クロッカス ) November 3 , 2012 December 15 , 2015 Natsu , Lucy , Gray
, Erza , and Wendy are chosen to represent their guild in the Grand Magic Games , which are to be held in Fiore 's capital , Crocus .
</item 7>

<item 8>
List of Fairy Tail episodes (season 6)
In the end , Cana of Fairy Tail Team B dominates the event by using the Fairy Glitter spell lent to her by Mavis , much to Makarov 's surprise . With Cana 's
victory , Fairy Tail Team B receives 8 points , and the audience began to burst once more into a resounding swarm of cheers and claps due to the finish of the
two Fairy Tail teams . Meanwhile , the Raven Tail team somehow managed to get an insider information regarding the battle matchups for the day , so the masked
leader Alexei tells his subordinates Flare , Obra , Nalpudding , and Kurohebi that it 's about time to proceed with their true mission . 168 `` Laxus vs. Alexei
'' `` Rakusasu vs. Arekusei '' ( ラクサス vs. アレクセイ ) February 9 , 2013 September 1 , 2015 In the battle portion of the Grand Magic Games ' third day , Milliana of
Mermaid Heel beats Semmes of Quatro Puppy while Saber Tooth 's Rufus swiftly defeats Blue Pegasus ' Eve .
</item 8>

<item 9>
List of Fairy Tail episodes (season 6)
Later , Jellal discusses the fact that he has not sensed the evil power similar to Zeref 's even though the second day of the Grand Magic Games has ended . Erza
reunites with Milliana , who is revealed to be a member of Mermaid Heel . Meanwhile , Saber Tooth 's master Jiemma is dissatisfied with his guild 's performance
. Though Sting is given another chance , Jiemma humiliates Yukino for her failure . 165 `` Hatred at Nightfall '' `` Urami wa Yoru no Tobari ni Tsutsumarete ''
( 怨み は 夜 の 帐 に 包 まれ て ) January 19 , 2013 September 1 , 2015 After Yukino is expelled from Saber Tooth , she offers her two golden Gatekeys to Lucy to complete
the latter 's collection of the Zodiac Gatekeys , stating that when the twelve Zodiac Gatekeys are all gathered , a gateway that changes the world will open ;
however , Lucy turns down the offer .
</item 9>

<item 10>
List of Fairy Tail episodes (season 6)
Meanwhile , the fourth day of the competitions comes to a start with an event , `` Naval Battle '' , which involves the participants swimming in a giant water
sphere and knocking the other competitors out . After the other participants fall out of the aquatic sphere , a five - minute battle between Lucy and Minerva
ensues . However , Minerva quickly attacks Lucy , and she manages to steal the keys of the Celestial Spirit mage . Regardless of this , Lucy refuses to
surrender . After hearing Lucy 's intentions , Minerva stops from attacking up to the extent that the five - minute mark was reached . However , Minerva resumes
attacking Lucy until the game officials decide to halt the match . The Fairy Tail members snap , and Minerva dangles Lucy out of the sphere while the others
rush to her aid .
</item 10>
=== END ITEM id=eace0b33d72b37ae ===

=== ITEM id=c76a04d92df7d814 ===
Gold sentences and facts:
Sentence 0: A regulatory site ( also known as an `` allosteric site '' ) is a site on an allosteric protein to which a modulator molecule ( or allosteric regulator ) binds . This ligand - binding site on a receptor or enzyme is distinct from the active site . Allosteric modulators alter enzyme activity by binding to the regulatory site . Some allosteric regulators activate enzymes , while others inhibit them . The working hypothesis is that when allosteric regulators bind to the regulatory site , the protein undergoes conformational changes which changes the enzyme 's activity .
  Fact 0.0: A regulatory site is also known as an allosteric site.
  Fact 0.1: A regulatory site is a site on an allosteric protein.
  Fact 0.2: A modulator molecule binds to a regulatory site on an allosteric protein.
  Fact 0.3: A modulator molecule is also known as an allosteric regulator.
  Fact 0.4: The ligand-binding site on a receptor or enzyme is distinct from the active site.
  Fact 0.5: Allosteric modulators alter enzyme activity by binding to the regulatory site.
  Fact 0.6: Some allosteric regulators activate enzymes.
  Fact 0.7: Other allosteric regulators inhibit enzymes.
  Fact 0.8: The working hypothesis is that when allosteric regulators bind to the regulatory site, the protein undergoes conformational changes.
  Fact 0.9: The working hypothesis is that these conformational changes change the enzyme's activity.

Retrieved items:
<item 1>
A regulatory, or allosteric, site is a protein location separate from its active site where a modulator binds. Binding can change the protein’s shape and enzyme
activity, either activating or inhibiting it. The article is an unsourced stub and has been proposed for merger with “Allosteric regulation.”
</item 1>

<item 2>
Regulatory site
This biochemistry article is a stub . You can help Wikipedia by expanding it . Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Regulatory_site&oldid=807494796 '' Categories : Protein structure Biochemistry stubs Hidden categories : Articles to
be merged from October 2017 All articles to be merged Articles lacking sources from December 2009 All articles lacking sources All stub articles Talk About
Wikipedia Add links This page was last edited on 28 October 2017 , at 09 : 38 . About Wikipedia
</item 2>

<item 3>
ATP synthase
5.5 - 6 : Polymerization motors Dynamin Tubulin Enzymes Activity Active site Binding site Catalytic triad Oxyanion hole Enzyme promiscuity Catalytically perfect
enzyme Coenzyme Cofactor Enzyme catalysis Regulation Allosteric regulation Cooperativity Enzyme inhibitor Classification EC number Enzyme superfamily Enzyme
family List of enzymes Kinetics Enzyme kinetics Eadie -- Hofstee diagram Hanes -- Woolf plot Lineweaver -- Burk plot Michaelis -- Menten kinetics Types EC1
Oxidoreductases ( list ) EC2 Transferases ( list ) EC3 Hydrolases ( list ) EC4 Lyases ( list ) EC5 Isomerases ( list ) EC6 Ligases ( list ) Molecular and
Cellular Biology portal Retrieved from `` https://en.wikipedia.org/w/index.php?title=ATP_synthase&oldid=821081799 '' Categories : Enzymes Cellular respiration
Photosynthesis EC 3.6.
</item 3>

<item 4>
ATP synthase is a conserved, reversible rotary enzyme: proton flow through membrane-embedded F₀ rotates a c-ring and central stalk, driving F₁’s three catalytic
β sites through conformations that synthesize ATP; ATP hydrolysis can instead generate a proton gradient. It functions in mitochondria, bacteria and
chloroplasts. Research covers its evolution, assembly, structure, catalytic residues, oligomycin sensitivity and rotary mechanism, including high-resolution
bovine structures. The material also
</item 4>

<item 5>
The cited studies examine ATP synthase evolution, including gene duplication, changes in H⁺/ATP coupling, and reversals between ATPase and synthase functions.
Other work investigates catalytic-site charged residues in *E. coli*, yeast mitochondrial enzyme assembly and function, and the rotary mechanism and
architecture of F-type ATP synthases. Structural studies report bovine mitochondrial F₁-ATPase at 2.8 Å resolution and its central stalk at 2.4 Å.
</item 5>

<item 6>
The cited research covers ATP synthase’s oligomycin sensitivity, H⁺-ATPases, mitochondrial structures revealed by cryo-EM, and the atomic model of its dimeric
Fₒ region. It also examines cooperative catalytic sites, supporting an alternating three-site model, and explains the enzyme’s rotary mechanism.
</item 6>

<item 7>
ATP synthase
`` Atomic model for the dimeric FO region of mitochondrial ATP synthase '' . Science . 358 ( 6365 ) : 936 -- 40 . doi : 10.1126 / science. aao4815 . PMID
29074581 . Jump up ^ Gresser MJ , Myers JA , Boyer PD ( October 1982 ) . `` Catalytic site cooperativity of beef heart mitochondrial F adenosine triphosphatase
. Correlations of initial velocity , bound intermediate , and oxygen exchange measurements with an alternating three - site model '' . The Journal of Biological
Chemistry . 257 ( 20 ) : 12030 -- 8 . PMID 6214554 . Jump up ^ Nakamoto RK , Baylis Scanlon JA , Al - Shawi MK ( August 2008 ) . `` The rotary mechanism of the
ATP synthase '' . Archives of Biochemistry and Biophysics. 476 ( 1 ) : 43 -- 50 . doi : 10.1016 / j. abb. 2008.05. 004 . PMC 2581510 .
</item 7>

<item 8>
Regulatory site
Regulatory site - Wikipedia Regulatory site Jump to : navigation , search It has been suggested that this article be merged with Allosteric regulation . (
Discuss ) Proposed since October 2017 . This article does not cite any sources . Please help improve this article by adding citations to reliable sources .
Unsourced material may be challenged and removed . ( December 2009 ) ( Learn how and when to remove this template message ) A regulatory site ( also known as an
`` allosteric site '' ) is a site on an allosteric protein to which a modulator molecule ( or allosteric regulator ) binds . This ligand - binding site on a
receptor or enzyme is distinct from the active site . Allosteric modulators alter enzyme activity by binding to the regulatory site . Some allosteric regulators
activate enzymes , while others inhibit them . The working hypothesis is that when allosteric regulators bind to the regulatory site , the protein undergoes
conformational changes which changes the enzyme 's activity . This biochemistry article is a stub . You can help Wikipedia by expanding it .
</item 8>

<item 9>
The Wikipedia article “Regulatory site” is a biochemistry stub categorized under protein structure. It was marked for merging in October 2017 and lacks sources;
the page was last edited on October 28, 2017.
</item 9>

<item 10>
ATP synthase
The cryo - EM model of ATP synthase suggests that the peripheral stalk is a flexible structure that wraps around the complex as it joins F to F. Under the right
conditions , the enzyme reaction can also be carried out in reverse , with ATP hydrolysis driving proton pumping across the membrane . The binding change
mechanism involves the active site of a β subunit 's cycling between three states . In the `` loose '' state , ADP and phosphate enter the active site ; in the
adjacent diagram , this is shown in pink . The enzyme then undergoes a change in shape and forces these molecules together , with the active site in the
resulting `` tight '' state ( shown in red ) binding the newly produced ATP molecule with very high affinity . Finally , the active site cycles back to the open
state ( orange ) , releasing ATP and binding more ADP and phosphate , ready for the next cycle of ATP production . Physiological role ( edit ) Like other
enzymes , the activity of F F ATP synthase is reversible .
</item 10>
=== END ITEM id=c76a04d92df7d814 ===

=== ITEM id=469a28c9d99c69b5 ===
Gold sentences and facts:
Sentence 0: `` Lord Have Mercy on the Working Man '' is a song written by Kostas and recorded by American country music singer Travis Tritt . It was released in August 1992 as the first of five singles from his third studio album , T-R-O-U-B-L-E . The song became Tritt 's tenth entry on the Billboard Hot Country Singles & Tracks ( now Hot Country Songs ) charts , where it peaked at number 5 .
  Fact 0.0: Lord Have Mercy on the Working Man is a song.
  Fact 0.1: Kostas wrote the song Lord Have Mercy on the Working Man.
  Fact 0.2: Travis Tritt, an American country music singer, recorded the song Lord Have Mercy on the Working Man.
  Fact 0.3: Lord Have Mercy on the Working Man was released in August 1992.
  Fact 0.4: Lord Have Mercy on the Working Man was the first of five singles from Travis Tritt's third studio album T-R-O-U-B-L-E.
  Fact 0.5: Lord Have Mercy on the Working Man became Travis Tritt's tenth entry on the Billboard Hot Country Singles & Tracks charts.
  Fact 0.6: The Billboard Hot Country Singles & Tracks chart is now called Hot Country Songs.
  Fact 0.7: Lord Have Mercy on the Working Man peaked at number 5 on the Billboard chart.
Sentence 1: `` Lord Have Mercy on the Working Man '' Single by Travis Tritt from the album T-R-O-U-B-L-E B - side `` Lord Have Mercy on the Working Man '' ( album version ) Released August 10 , 1992 Format CD single Genre Country Length 4 : 55 ( album version ) Label Warner Bros. Nashville Songwriter ( s ) Kostas Producer ( s ) Gregg Brown Travis Tritt singles chronology `` Nothing Short of Dying '' ( 1992 ) `` Lord Have Mercy on the Working Man '' ( 1992 ) `` Can I Trust You with My Heart '' ( 1992 ) `` Nothing Short of Dying '' ( 1992 ) ' `` Lord Have Mercy on the Working Man '' ' ( 1992 ) `` Can I Trust You with My Heart '' ( 1992 )
  Fact 1.0: Lord Have Mercy on the Working Man is a single by Travis Tritt from the album T-R-O-U-B-L-E.
  Fact 1.1: The B-side of the single is Lord Have Mercy on the Working Man (album version).
  Fact 1.2: The single was released on August 10, 1992.
  Fact 1.3: The single format is CD single.
  Fact 1.4: The genre of the single is country.
  Fact 1.5: The album version of the song has a length of 4:55.
  Fact 1.6: The label of the single is Warner Bros. Nashville.
  Fact 1.7: Kostas is the songwriter of Lord Have Mercy on the Working Man.
  Fact 1.8: Gregg Brown is the producer of Lord Have Mercy on the Working Man.
  Fact 1.9: The Travis Tritt singles chronology lists Nothing Short of Dying (1992), Lord Have Mercy on the Working Man (1992), and Can I Trust You with My Heart (1992).
  Fact 1.10: The Travis Tritt singles chronology then repeats Nothing Short of Dying (1992), Lord Have Mercy on the Working Man (1992), and Can I Trust You with My Heart (1992).

Retrieved items:
<item 1>
Lord Have Mercy on the Working Man
It was released in August 1992 as the first of five singles from his third studio album , T-R-O-U-B-L-E . The song became Tritt 's tenth entry on the Billboard
Hot Country Singles & Tracks ( now Hot Country Songs ) charts , where it peaked at number 5 . Contents ( hide ) 1 Content 2 Critical reception 3 Music video 4
Chart performance 5 References 6 External links Content ( edit ) `` Lord Have Mercy on the Working Man '' is a moderate up - tempo whose lyrics centralize on a
theme of economic injustice towards blue collar workers . The instrumentation features various forms of percussion from Sam Bacco , including crotales , wobble
board , spoons and a broom . Richard Bennett and Wendell Cox play guitar solos before the third verse , and Brooks & Dunn , T. Graham Brown , George Jones ,
Little Texas , Dana McVicker , Tanya Tucker and Porter Wagoner all sing background vocals on the final chorus .
</item 1>

<item 2>
What a Friend We Have in Jesus
The tune was in the labor protest movement due to its wide popularity . `` Dump the Bosses Off Your Backs '' ( words by John Brill ) was published in the IWW ``
Little Red Songbook '' and has been recorded by dozens of artists including Utah Phillips , Anne Feeney , and Joe Glazer . A Norwegian version was used in the
film Trollhunter . Published versions ( edit ) Rise Up Singing , page 98 The Lutheran Hymnal , hymn # 457 References ( edit ) Wikisource has original text
related to this article : What a Friend We Have in Jesus Jump up ^ `` What a Friend We Have in Jesus '' . Retrieved 2007 - 05 - 03 . Jump up ^ Polack , W.G. (
1941 ) . Handbook to the Lutheran Hymnal . St. Louis : Concordia . p. 323 . Jump up ^ `` Washington Phillips discography '' . wirz.de . Retrieved August 24 ,
2015 . Jump up ^ George Washington Phillips : Jesus Is My Friend at AllMusic .
</item 2>

<item 3>
Lord Have Mercy on the Working Man
Ai n't Country '' My Honky Tonk History `` The Girl 's Gone Wild '' `` What Say You '' ( with John Mellencamp ) `` I See Me '' The Storm `` You Never Take Me
Dancing '' `` Something Stronger Than Me '' The Calm After . `` Sometimes Love Just Ai n't Enough '' ( With Tyler Reese ) Other songs `` This One 's Gonna Hurt
You ( For a Long , Long Time ) '' ( with Marty Stuart ) `` Take It Easy '' `` The Devil Comes Back to Georgia '' ( With Mark O'Connor , Charlie Daniels , Johnny
Cash , and Marty Stuart ) `` Honky Tonkin 's What I Do Best '' ( With Marty Stuart ) `` Hope '' `` Here 's Your Sign ( Get the Picture ) '' ( With Bill Engvall
) `` Same Old Train '' `` Out of Control Raging Fire '' ( with Patty Loveless ) `` Southern Boy '' ( with Charlie Daniels ) `` Bible Belt '' ( with Little Feat
) `` Move It On Over '' `` Lonesome , On'ry and Mean '' Retrieved from `` https://en.wikipedia.org/w/index.
</item 3>

<item 4>
Lord Have Mercy on the Working Man
Dying '' T-R-O-U-B-L-E `` Lord Have Mercy on the Working Man '' `` Can I Trust You with My Heart '' `` T-R-O-U-B-L-E '' `` Looking Out for Number One '' ``
Worth Every Mile '' Ten Feet Tall and Bulletproof `` Foolish Pride '' `` Ten Feet Tall and Bulletproof '' `` Between an Old Memory and Me '' `` Tell Me I Was
Dreaming '' Greatest Hits : From the Beginning `` Sometimes She Forgets '' `` Only You ( And You Alone ) '' The Restless Kind `` More Than You 'll Ever Know ''
`` Where Corn Do n't Grow '' `` She 's Going Home with Me '' `` Helping Me Get Over You '' ( With Lari White ) `` Still in Love with You '' No More Looking over
My Shoulder `` If I Lost You '' `` No More Looking Over My Shoulder '' `` Start the Car '' Down the Road I Go `` Best of Intentions '' `` It 's a Great Day to
Be Alive '' `` Love of a Woman '' `` Modern Day Bonnie and Clyde '' Strong Enough `` Strong Enough to Be Your Man '' `` Country Ai n't Country '' My Honky Tonk
History `` The Girl 's Gone Wild '' ``
</item 4>

<item 5>
Lord Have Mercy on the Working Man
Music video ( edit ) Jack Cole directed the song 's music video . In the book Country Music Culture : From Hard Times to Heaven , Curtis W. Ellison wrote that
the song 's music video `` confronted a litany of personal oppressions attributed to government policy '' that coincided with Bill Clinton 's presidential
campaign . Chart performance ( edit ) `` Lord Have Mercy on the Working Man '' spent 20 weeks on the Billboard Hot Country Singles & Tracks ( now Hot Country
Songs ) charts in 1992 , peaking at number 5 . The song also reached number 10 on the Canadian RPM Country Tracks charts . Chart ( 1992 ) Peak position Canada
Country Tracks ( RPM ) 10 US Hot Country Songs ( Billboard ) 5 References ( edit ) ^ Jump up to : T-R-O-U-B-L-E ( cassette insert ) . Travis Tritt . Warner
Bros. Records . 1992 . 45048 . Jump up ^ Hurst , Jack ( 13 August 1992 ) . `` Satellite show Travis Tritt to launch new album from Park West '' . Chicago
Tribune . Retrieved 28 August 2009 .
</item 5>

<item 6>
Lord Have Mercy on the Working Man
Chicago Tribune . Retrieved 28 August 2009 . ^ Jump up to : Whitburn , Joel ( 2006 ) . The Billboard Book of Top 40 Country Hits ( 2 ed . ) . Billboard Books .
p. 353 . ISBN 0 - 8230 - 8291 - 1 . Jump up ^ Walker , Teresa M. ( 4 September 1992 ) . `` Travis Tritt 's rockin ' country '' . Gainesville Sun . Archived from
the original on 12 July 2012 . Retrieved 28 August 2009 . Jump up ^ Larsen , Dave ( 18 September 1992 ) . `` Recordings on review '' . Dayton Daily News .
Retrieved 28 August 2009 . Jump up ^ Nash , Alanna ( 4 September 1992 ) . `` T-R-O-U-B-L-E in mind '' . Entertainment Weekly . Retrieved 28 August 2009 . Jump
up ^ Billboard , August 15 , 1992 Jump up ^ Ellison , Curtis W.
</item 6>

<item 7>
Lord Have Mercy on the Working Man
On'ry and Mean '' Retrieved from `` https://en.wikipedia.org/w/index.php?title=Lord_Have_Mercy_on_the_Working_Man&oldid=789364103 '' Categories : 1992 singles
Travis Tritt songs Songs written by Kostas ( songwriter ) Warner Bros. Records singles 1992 songs Hidden categories : Articles with hAudio microformats
Singlechart usages for Canadacountry Singlechart called without artist Singlechart called without song Singlechart usages for Billboardcountrysongs Talk
Contents About Wikipedia Add links This page was last edited on 6 July 2017 , at 22 : 41 . Text is available under the Creative Commons Attribution - ShareAlike
License ; additional terms may apply . By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the
Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 7>

<item 8>
Lord Have Mercy on the Working Man
( 1995 ) . Country Music Culture : From Hard Times to Heaven . Univ . Press of Mississippi . p. 233 . ISBN 0 - 87805 - 722 - 6 . Retrieved 28 August 2009 . Jump
up ^ `` Top RPM Country Tracks : Issue 1864 . '' RPM . Library and Archives Canada . November 28 , 1992 . Retrieved August 15 , 2013 . Jump up ^ `` Travis Tritt
Chart History ( Hot Country Songs ) '' . Billboard . External links ( edit ) Lyrics of this song at MetroLyrics ( hide ) Travis Tritt singles Country Club ``
Country Club '' `` Help Me Hold On '' `` I 'm Gonna Be Somebody '' `` Put Some Drive in Your Country '' `` Drift Off to Dream '' It 's All About to Change ``
Here 's a Quarter ( Call Someone Who Cares ) '' `` Anymore '' `` The Whiskey Ai n't Workin ' '' ( With Marty Stuart ) `` Nothing Short of Dying '' T-R-O-U-B-L-E
`` Lord Have Mercy on the Working Man '' ``
</item 8>

<item 9>
Lord Have Mercy on the Working Man
Lord Have Mercy on the Working Man - wikipedia Lord Have Mercy on the Working Man Jump to : navigation , search `` Lord Have Mercy on the Working Man '' Single
by Travis Tritt from the album T-R-O-U-B-L-E B - side `` Lord Have Mercy on the Working Man '' ( album version ) Released August 10 , 1992 Format CD single
Genre Country Length 4 : 55 ( album version ) Label Warner Bros. Nashville Songwriter ( s ) Kostas Producer ( s ) Gregg Brown Travis Tritt singles chronology ``
Nothing Short of Dying '' ( 1992 ) `` Lord Have Mercy on the Working Man '' ( 1992 ) `` Can I Trust You with My Heart '' ( 1992 ) `` Nothing Short of Dying '' (
1992 ) ' `` Lord Have Mercy on the Working Man '' ' ( 1992 ) `` Can I Trust You with My Heart '' ( 1992 ) `` Lord Have Mercy on the Working Man '' is a song
written by Kostas and recorded by American country music singer Travis Tritt .
</item 9>

<item 10>
Lord Have Mercy on the Working Man
Critical reception ( edit ) Teresa M. Walker , in her review for the Gainesville Sun , said that with the assistance from superstars on the final chorus , the
song `` should shoot up the charts . '' Dave Larsen of the Dayton Daily News cited it as one of the stronger tracks on the album , saying that the album ``
works best when Tritt sticks with the populist approach . '' Alanna Nash of Entertainment Weekly said of the song , `` Tritt finds a solid image for his laborer
's lament ( ... ) But the melody is so slight that he resorts to an acoustic arrangement that evokes Jimmie Rodgers and the young Roy Acuff . '' Deborah Evans
Price , of Billboard magazine gave the song a mixed review , saying that while the final chorus of song features famous country artists , it was too bad that
the entire song is n't as strong as the last verse . Music video ( edit ) Jack Cole directed the song 's music video .
</item 10>
=== END ITEM id=469a28c9d99c69b5 ===

=== ITEM id=a35e7b97b12ad392 ===
Gold sentences and facts:
Sentence 0: The Eagles headed into their bye week with the best record in the NFL ( 8 -- 1 ) , and with Wentz leading the NFL MVP discussion with an NFL - leading 23 touchdowns to 5 interceptions . During their bye , the Eagles signed former Giants offensive tackle Will Beatty and former Saints linebacker Dannell Ellerbe to not only add depth but also players with Super Bowl experiences . Coming off of the bye week , the Philadelphia Eagles entered a Sunday night showdown with the defending NFC East champion Dallas Cowboys . The Eagles came out sluggish and trailed 9 -- 7 at the half . The team exploded in the second half though , rattling off 30 unanswered points to soundly defeat their rival by a final score of 37 -- 9 . The Eagles were rolling on a nine - game winning streak after a huge blowout win over the Chicago Bears . The Eagles traveled to Seattle in Week 13 in a huge Sunday Night showdown against Russell Wilson and the Seattle Seahawks . The Eagles struggled throughout the whole game , blowing many scoring opportunities and allowing Wilson to throw 3 touchdowns . The Eagles lost 24 -- 10 , ending their win streak . However , the Eagles bounced back in Week 14 win over the Los Angeles Rams in a tough 43 -- 35 win . However , Carson Wentz left the game in the third quarter with a knee injury . It was later found out that Wentz tore his ACL , ending his MVP caliber season . Backup Nick Foles would once again take over as starting quarterback . In Foles 's first start against the New York Giants , he would lead them back from a 20 -- 7 deficit and score 4 touchdowns and win the game 34 -- 29 . Foles struggled in the last two games of the season against the Oakland Raiders and the Dallas Cowboys , and threw a touchdown and 2 interceptions in that span . Despite this , the Eagles finished 13 -- 3 and secured the top seed in the NFC . The Eagles opened as the underdogs , the first time in history that a No. 1 seed has opened up the postseason as an underdog . Despite this , Foles would lead the Eagles past the Atlanta Falcons in the Divisional Round 15 -- 10 . In the next game Foles and the Eagles annihilated the Minnesota Vikings 38 -- 7 in the NFC Championship nicknamed the Minneapolis Massacre , mocking the Minneapolis Miracle from their previous playoff victory . Foles had his best game since week 15 and threw for 352 passing yards and 3 touchdowns including an impressive flea flicker touchdown pass . This win marked the franchise 's third Super Bowl appearance and a berth in Super Bowl LII for a 2005 rematch against Tom Brady and the New England Patriots .
  Fact 0.0: The Eagles headed into their bye week with the best record in the NFL, at 8-1.
  Fact 0.1: Carson Wentz was leading the NFL MVP discussion.
  Fact 0.2: Wentz had an NFL-leading 23 touchdowns.
  Fact 0.3: Wentz had 5 interceptions.
  Fact 0.4: During their bye, the Eagles signed former Giants offensive tackle Will Beatty.
  Fact 0.5: During their bye, the Eagles signed former Saints linebacker Dannell Ellerbe.
  Fact 0.6: The signings added depth to the Eagles.
  Fact 0.7: Beatty and Ellerbe have Super Bowl experience.
  Fact 0.8: Coming off the bye week, the Eagles entered a Sunday night showdown with the defending NFC East champion Dallas Cowboys.
  Fact 0.9: The Eagles came out sluggish against the Cowboys and trailed 9-7 at the half.
  Fact 0.10: In the second half, the Eagles scored 30 unanswered points.
  Fact 0.11: The Eagles defeated the Cowboys 37-9.
  Fact 0.12: The Eagles were on a nine-game winning streak.
  Fact 0.13: The nine-game winning streak came after a huge blowout win over the Chicago Bears.
  Fact 0.14: The Eagles traveled to Seattle in Week 13 for a Sunday Night showdown against Russell Wilson and the Seattle Seahawks.
  Fact 0.15: The Eagles struggled throughout that game, blowing many scoring opportunities.
  Fact 0.16: Wilson threw 3 touchdowns in that game.
  Fact 0.17: The Eagles lost to Seattle 24-10.
  Fact 0.18: The loss ended the Eagles' win streak.
  Fact 0.19: The Eagles bounced back in Week 14 with a tough 43-35 win over the Los Angeles Rams.
  Fact 0.20: Carson Wentz left the game in the third quarter with a knee injury.
  Fact 0.21: It was later found that Wentz tore his ACL.
  Fact 0.22: The ACL tear ended Wentz's MVP-caliber season.
  Fact 0.23: Backup Nick Foles would once again take over as starting quarterback.
  Fact 0.24: In Foles's first start against the New York Giants, he led the Eagles back from a 20-7 deficit.
  Fact 0.25: In that game, Foles scored 4 touchdowns.
  Fact 0.26: The Eagles won that game 34-29.
  Fact 0.27: Foles struggled in the last two games of the season against the Oakland Raiders and the Dallas Cowboys.
  Fact 0.28: Across those two games, Foles threw a touchdown and 2 interceptions.
  Fact 0.29: Despite this, the Eagles finished 13-3.
  Fact 0.30: The Eagles secured the top seed in the NFC.
  Fact 0.31: The Eagles opened the postseason as underdogs.
  Fact 0.32: It was the first time in history that a No. 1 seed opened the postseason as an underdog.
  Fact 0.33: Foles led the Eagles past the Atlanta Falcons 15-10 in the Divisional Round.
  Fact 0.34: In the next game, Foles and the Eagles annihilated the Minnesota Vikings 38-7 in the NFC Championship.
  Fact 0.35: That NFC Championship game was nicknamed the Minneapolis Massacre.
  Fact 0.36: The nickname mocked the Minneapolis Miracle from the Eagles' previous playoff victory.
  Fact 0.37: Foles had his best game since week 15 in the NFC Championship.
  Fact 0.38: Foles threw for 352 passing yards in that game.
  Fact 0.39: Foles threw 3 touchdowns in that game, including an impressive flea flicker touchdown pass.
  Fact 0.40: The win marked the franchise's third Super Bowl appearance.
  Fact 0.41: The win gave the Eagles a berth in Super Bowl LII.
  Fact 0.42: Super Bowl LII was a rematch of 2005 against Tom Brady and the New England Patriots.
Sentence 1: Playoff record ( all - time ) : 22 -- 21 ( last appearance after beating New England Patriots in Super Bowl LII )
  Fact 1.0: The Eagles' all-time playoff record is 22-21.
  Fact 1.1: The record is as of their last appearance, after beating the New England Patriots in Super Bowl LII.

Retrieved items:
<item 1>
Super Bowl XXXIX
The Eagles failed to recover their ensuing onside kick attempt , with Christian Fauria catching the ball cleanly and sliding down to seal possession for New
England . The Patriots then played it safe by running the ball 3 times and forcing Philadelphia to use all of its timeouts . New England punter Josh Miller then
pinned the Eagles back at their own 4 - yard line with just 46 seconds left in the game . Philadelphia then tried one last desperate drive to win or tie the
game . But on first down , McNabb was pressured into making a rushed pass to Westbrook at the line of scrimmage . Instead of dropping the pass to stop the clock
, Westbrook made the mistake of catching the ball and was immediately tackled for no gain , keeping the clock running and forcing the Eagles to run back to the
line of scrimmage for their next play with no huddle . On second down , McNabb threw an incomplete pass intended for Owens . Finally on third down , McNabb
threw a pass that deflected off of the outstretched fingertips of Smith and into the arms of Harrison for an interception with nine seconds left .
</item 1>

<item 2>
Super Bowl XXXIX
For the Eagles , with the loss , the city of Philadelphia would not see its first championship since the 76ers swept the 1983 NBA Finals until the next time
their city 's teams played for a championship , when the Phillies won the 2008 World Series . The Eagles did not return to the Super Bowl until 2017 , also with
a 13 - 3 record , and defeating the Falcons and Vikings in the playoffs . The Eagles and the Patriots met again in Super Bowl LII , following the 2017 season ,
with the Eagles taking their revenge 41 -- 33 . This is the last time that a team has won back - to - back Super Bowls , with the Patriots losing their second
playoff game a year later and then , for the next eight years , every team either losing their first playoff game or missing them altogether . The streak was
broken in 2014 when the Seahawks defeated the Panthers and ultimately advanced to the Super Bowl , only to lose to a Patriots goal line stand .
</item 2>

<item 3>
Philadelphia Eagles
Foles had his best game since week 15 and threw for 352 passing yards and 3 touchdowns including an impressive flea flicker touchdown pass . This win marked the
franchise 's third Super Bowl appearance and a berth in Super Bowl LII for a 2005 rematch against Tom Brady and the New England Patriots . With Foles at the
helm , the Eagles started off the game strong , leading the Patriots 22 -- 12 at halftime . In the second half the Patriots gained momentum and for the first
time took the lead in the 4th quarter 33 -- 32 . The Eagles rallied back and scored an 11 - yard Touchdown to tight end Zach Ertz . Later the Eagles converted
on a 46 - yard field goal by Jake Elliott to lead 41 -- 33 . On the final play of the game with 9 seconds left , Brady threw a hail - mary pass directed toward
tight end Rob Gronkowski which was bobbled by Patrick Robinson before being dropped in the end - zone as time expired , securing the win for the Eagles 41 -- 33
, and ending the 58 - year long championship drought .
</item 3>

<item 4>
Super Bowl XXXIX
Super Bowl XXXIX - wikipedia Super Bowl XXXIX Jump to : navigation , search Super Bowl XXXIX New England Patriots ( AFC ) Philadelphia Eagles ( NFC ) 24 21
Total NE 0 7 7 10 24 PHI 0 7 7 7 21 Date February 6 , 2005 ( 2005 - 02 - 06 ) Stadium Alltel Stadium , Jacksonville , Florida MVP Deion Branch , Wide Receiver
Favorite Patriots by 7 Referee Terry McAulay Attendance 78,125 Future Hall of Famers Patriots : None Eagles : Brian Dawkins , Terrell Owens Ceremonies National
anthem The combined choirs of the U.S. Military Academy , the U.S. Naval Academy , the U.S. Air Force Academy , and the U.S. Coast Guard Academy , and U.S. Army
Herald Trumpets .
</item 4>

<item 5>
Super Bowl XXXIX
The next year , the Patriots , having faltered down the stretch , entered those playoffs as the second seed behind the Broncos , who then defeated New England
in the AFC Title Game that year en route to a title of their own in Super Bowl 50 . Two years later , the Patriots , defending champions themselves from Super
Bowl LI , returned to repeat their title only to lose the aforementioned Super Bowl LII . Final statistics ( edit ) Sources : NFL.com Super Bowl XXXIX , Super
Bowl XXXIX Play Finder NE , Super Bowl XXXIX Play Finder Phi Statistical comparison ( edit ) New England Patriots Philadelphia Eagles First downs 21 24 First
downs rushing 6 First downs passing 14 18 First downs penalty Third down efficiency 4 / 12 9 / 16 Fourth down efficiency 0 / 0 0 / 0 Net yards rushing 112 45
Rushing attempts 28 17 Yards per rush 4.0 2.
</item 5>

<item 6>
Philadelphia Eagles
Season records Main article : List of Philadelphia Eagles seasons Regular season record ( all - time ) : 568 -- 594 -- 26 Playoff record ( all - time ) : 22 --
21 ( last appearance after beating New England Patriots in Super Bowl LII ) Most points in a season : 474 points ( 2014 ) NFL championships won : 4 ( 3 before
the 1967 NFL - AFL merger created the Super Bowl ) Super Bowls won : 1 out of 3 appearances Passing leader ( all - time ) : Donovan McNabb -- 32,873 yards
Rushing leader ( all - time ) : LeSean McCoy -- 6,792 yards Receiving leader ( all - time ) : Harold Carmichael -- 8,978 yards Winningest coach ( all - time ) :
Andy Reid -- 130 wins Top player by approximate value ( all - time ) : Donovan McNabb -- 126 AV Rivalries New York Giants Main article : Eagles -- Giants
rivalry One of the NFL 's oldest , this rivalry began on October 15 , 1933 when the Giants defeated the newly founded Eagles 56 -- 0 .
</item 6>

<item 7>
Super Bowl XXXIX
It was the first time New England trailed during the entire postseason . On their ensuing drive , the Patriots moved the ball to the Eagles four - yard line ,
mainly on plays by running back Corey Dillon , who caught two screen passes for 29 yards and rushed for 25 . But quarterback Tom Brady fumbled the ball on a
fake handoff Play - action pass and Philadelphia defender Darwin Walker recovered it . However , the Eagles could not take advantage of the turnover and had to
punt after 3 plays . Eagles punter Dirk Johnson 's punt went just 29 yards , giving the Patriots the ball at Philadelphia 's 37 - yard line . The Patriots then
drove 37 yards to score on Brady 's 4 - yard pass to receiver David Givens with 1 : 10 remaining in the period , tying the game 7 -- 7 by halftime . It was only
the second halftime tie in Super Bowl history and the first time both of the game 's first two quarters ended tied .
</item 7>

<item 8>
Super Bowl XXXIX
The Patriots , who entered the Super Bowl after compiling a 14 -- 2 regular season record , became the first team since the 1997 -- 1998 Denver Broncos to win
consecutive Super Bowls . New England also became the second team after the Dallas Cowboys to win three Super Bowls in four years . The Eagles were making their
second Super Bowl appearance after posting a 13 -- 3 regular season record . The game was close throughout , with the teams battling to a 14 -- 14 tie by the
end of the third quarter . The Patriots then scored 10 points in the 4th quarter with Corey Dillon 's 2 - yard touchdown run and Adam Vinatieri 's 22 - yard
field goal . The Eagles then cut their deficit to 24 -- 21 , with quarterback Donovan McNabb 's 30 - yard touchdown pass to receiver Greg Lewis , with 1 : 48
remaining in the game but could not sustain the comeback . Overall , New England forced four turnovers , while Patriots wide receiver Deion Branch was named
Super Bowl MVP for recording 133 receiving yards and tied the Super Bowl record with 11 catches .
</item 8>

<item 9>
Super Bowl XXXIX
Tom Brady took a knee to run out the clock , clinching the 3rd Super Bowl title in 4 years for the Patriots , and in the eyes of many establishing themselves as
a dynasty . Adam Vinatieri warms up before the game An Eagles offensive play Philadelphia kicks off after scoring a touchdown Brian Westbrook 's touchdown catch
( photo 1 of 2 ) Brian Westbrook 's touchdown catch ( photo 2 of 2 ) Box score ( edit ) Super Bowl XXXIX : New England Patriots 24 , Philadelphia Eagles 21
Total Patriots ( AFC ) 0 7 7 10 24 Eagles ( NFC ) 0 7 7 7 21 at Alltel Stadium , Jacksonville , Florida Date : February 6 , 2005 Game time : 6 : 38 p.m. EST
Game weather : 59 ° F ( 15 ° C ) , clear ( hide ) Scoring summary Quarter Time Drive Team Scoring information Score Plays Yards TOP NE PHI 9 : 55 9 81 4 : 36
PHI L.J.
</item 9>

<item 10>
Philadelphia Eagles
Philadelphia Eagles . September 9 , 2016 . Retrieved September 17 , 2017 . Jump up ^ `` Philadelphia Eagles Team Capsule '' ( PDF ) . 2017 Official National
Football League Record and Fact Book . National Football League . August 22 , 2017 . Retrieved September 17 , 2017 . Jump up ^ `` Top 10 NFL Rivalries Of All
Time : No. 4 Giants - Eagles '' . Sports Illustrated . December 15 , 2005 . Retrieved September 17 , 2017 . Jump up ^ Chadiha , Jeffri ( October 31 , 2007 ) .
`` Ranking the NFL 's best rivalries : Where does Colts - Pats fit ? '' . ESPN.com . Retrieved April 12 , 2008 . Jump up ^ Bryan , Dave ( September 20 , 2016 )
. `` After 8 Straight Losses , Steelers Looking For Philadelphia Freedom Sunday Against Eagles '' . SteelersDepot.com . Retrieved September 17 , 2017 .
</item 10>
=== END ITEM id=a35e7b97b12ad392 ===
